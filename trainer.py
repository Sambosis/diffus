import torch
import torch.nn as nn
import threading
import queue
import traceback

# Internal imports
import config
from data_loader import get_dataloader, get_tokenizer
from diffusion_model import TextDiffusionModel


def _extract(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: tuple) -> torch.Tensor:
    """
    Extracts values from a 1D array `arr` at the indices specified by `timesteps`,
    and reshapes the output to a specified `broadcast_shape` for broadcasting.

    This is a standard utility function in diffusion model implementations.

    Args:
        arr (torch.Tensor): The 1D array to extract from (e.g., alphas_cumprod).
        timesteps (torch.Tensor): A tensor of timesteps (indices).
        broadcast_shape (tuple): The shape to broadcast the output to (e.g., (batch_size, 1, 1)).

    Returns:
        torch.Tensor: The extracted and reshaped tensor on the correct device.
    """
    res = arr.gather(-1, timesteps)
    # Reshape to (batch_size, 1, 1, ...) for broadcasting with tensors of shape (batch, seq, embed_dim)
    return res.reshape(broadcast_shape)


class TrainerThread(threading.Thread):
    """
    A separate thread for handling the model training process to prevent freezing the GUI.

    Manages the diffusion process (forward and reverse), the training loop, and sampling.
    Communicates progress and results back to the main GUI thread via a queue.
    """

    def __init__(self, update_queue: queue.Queue):
        """
        Initializes the TrainerThread.

        Args:
            update_queue (queue.Queue): A queue to send updates (stats, samples) to the main GUI thread.
        """
        super().__init__()
        self.daemon = True  # Allows main thread to exit even if this thread is running
        self.update_queue = update_queue
        self.running = False
        self.device = torch.device(config.DEVICE)

        # Components will be initialized by _initialize_components
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.loss_fn: nn.Module | None = None
        self.dataloader: torch.utils.data.DataLoader | None = None
        self.tokenizer = None  # Tokenizer instance

        # Diffusion schedule tensors will be initialized by _initialize_diffusion_schedule
        self.betas = None
        self.alphas = None
        self.alphas_cumprod = None
        self.alphas_cumprod_prev = None
        self.sqrt_alphas_cumprod = None
        self.sqrt_one_minus_alphas_cumprod = None
        self.posterior_variance = None

    def _initialize_components(self):
        """Initializes tokenizer, dataloader, model, and optimizer."""
        self._put_log("Initializing Tokenizer...")
        self.tokenizer = get_tokenizer()

        self._put_log("Initializing Dataloader...")
        self.dataloader = get_dataloader(tokenizer=self.tokenizer, batch_size=config.BATCH_SIZE)

        self._put_log("Initializing Model...")
        self.model = TextDiffusionModel().to(self.device)

        self._put_log("Initializing Optimizer...")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def _initialize_diffusion_schedule(self):
        """Pre-calculates the diffusion schedule (betas, alphas, etc.)."""
        self._put_log("Calculating diffusion schedule...")
        # Linear variance schedule as per the DDPM paper
        self.betas = torch.linspace(config.BETA_START, config.BETA_END, config.TIMESTEPS, device=self.device)
        
        # Alphas for forward process q(x_t | x_{t-1})
        self.alphas = 1. - self.betas
        # Cumulative product of alphas (alpha_bar) for q(x_t | x_0)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        # Previous cumulative product, padded at the start with 1.0 for t=0
        self.alphas_cumprod_prev = nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # --- Pre-calculated values for the forward process (q(x_t | x_0)) ---
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # --- Pre-calculated values for the reverse process p(x_{t-1} | x_t, x_0) ---
        # This is the variance of the posterior distribution q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self._put_log("Trainer initialization complete.")

    def forward_process(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Applies noise to the input embeddings `x_start` for a given timestep `t`.
        This implements the closed-form formula: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise.

        Args:
            x_start (torch.Tensor): The initial clean embeddings (batch_size, seq_len, embed_dim).
            t (torch.Tensor): The timesteps for each sample in the batch (batch_size,).
            noise (torch.Tensor): The noise to add, with the same shape as x_start.

        Returns:
            torch.Tensor: The noisy embeddings x_t.
        """
        broadcast_shape = (x_start.shape[0],) + (1,) * (len(x_start.shape) - 1)
        
        sqrt_alphas_cumprod_t = _extract(self.sqrt_alphas_cumprod, t, broadcast_shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, broadcast_shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def train_step(self, batch: dict) -> float:
        """
        Performs a single training step: forward process, model prediction, loss calculation, and backpropagation.

        Args:
            batch (dict): A batch from the dataloader, containing 'input_ids'.

        Returns:
            float: The loss value for this step.
        """
        self.optimizer.zero_grad()
        
        batch_token_ids = batch['input_ids'].to(self.device)
        
        # 1. Get initial embeddings from token IDs using the model's own embedding layer
        x_start = self.model.token_embedding(batch_token_ids)

        # 2. Sample random timesteps for each item in the batch
        t = torch.randint(0, config.TIMESTEPS, (batch_token_ids.shape[0],), device=self.device).long()
        
        # 3. Sample noise and apply the forward process to get noisy embeddings x_t
        noise = torch.randn_like(x_start)
        x_t = self.forward_process(x_start, t, noise)
        
        # 4. Get the model's prediction of the noise added to x_t
        predicted_noise = self.model(x_t, t)
        
        # 5. Calculate loss (MSE between actual noise and predicted noise) and update weights
        loss = self.loss_fn(noise, predicted_noise)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Gradient clipping
        self.optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def sample(self) -> str:
        """
        Generates a text sample using the reverse diffusion process (DDPM sampling).

        Returns:
            str: The decoded, generated text sentence.
        """
        self.model.eval()
        
        shape = (1, config.MAX_SEQ_LEN, config.EMBED_DIM)
        # Start with pure random noise (the state at T)
        x_t = torch.randn(shape, device=self.device)
        
        # Iteratively denoise the data from T-1 down to 0
        for t_val in reversed(range(0, config.TIMESTEPS)):
            t = torch.full((1,), t_val, device=self.device, dtype=torch.long)
            
            predicted_noise = self.model(x_t, t)

            # Use helper to extract schedule values for the current timestep t
            broadcast_shape = (x_t.shape[0],) + (1,) * (len(x_t.shape) - 1)
            alpha_t = _extract(self.alphas, t, broadcast_shape)
            beta_t = _extract(self.betas, t, broadcast_shape)
            sqrt_one_minus_alpha_cumprod_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, broadcast_shape)

            # Equation from DDPM paper to get the mean of the posterior p(x_{t-1} | x_t)
            model_mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)
            
            if t_val == 0:
                # At the final step, no more noise is added
                x_t = model_mean
            else:
                posterior_variance_t = _extract(self.posterior_variance, t, broadcast_shape)
                noise = torch.randn_like(x_t)
                # Add noise to get the sample for the previous timestep
                x_t = model_mean + torch.sqrt(posterior_variance_t) * noise

        # The final denoised output x_0 is an embedding
        denoised_embedding = x_t.squeeze(0)  # Shape: [seq_len, embed_dim]
        
        # Convert embeddings back to token IDs by finding the nearest neighbor in the vocab
        vocab_embeddings = self.model.token_embedding.weight.data
        # Calculate pairwise L2 distance: [seq_len, embed_dim] vs [vocab_size, embed_dim]
        dist = torch.cdist(denoised_embedding, vocab_embeddings, p=2)  # Shape: [seq_len, vocab_size]
        token_ids = torch.argmin(dist, dim=1).cpu().tolist()
        
        generated_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        self.model.train() # Set model back to training mode
        return generated_text

    def run(self):
        """The main entry point for the thread. Initializes components and runs the training loop."""
        try:
            self._initialize_components()
            self._initialize_diffusion_schedule()
        except Exception:
            error_msg = f"Error during trainer initialization:\n{traceback.format_exc()}"
            self._put_error(error_msg)
            return # Abort if initialization fails

        self.running = True
        step = 0
        self._put_log("Starting training loop...")
        try:
            for epoch in range(1, config.EPOCHS + 1):
                if not self.running: break
                
                for i, batch in enumerate(self.dataloader):
                    if not self.running: break
                    
                    loss = self.train_step(batch)
                    
                    if step % config.LOG_INTERVAL == 0:
                        self.update_queue.put({"type": "stats", "epoch": epoch, "step": step, "loss": loss})

                    if step % config.SAMPLE_INTERVAL == 0 and step > 0:
                        self._put_log(f"Step {step}: Generating sample...")
                        sample_text = self.sample()
                        self.update_queue.put({"type": "sample", "text": sample_text})
                    
                    step += 1

            if self.running: # If not stopped manually
                self._put_log("Training finished.")
                self.update_queue.put({"type": "done"})

        except Exception:
            error_msg = f"Error in training loop:\n{traceback.format_exc()}"
            self._put_error(error_msg)
        finally:
            self.running = False

    def stop(self):
        """Stops the training loop gracefully."""
        if self.running:
            self._put_log("Stopping trainer...")
            self.running = False

    def _put_log(self, message: str):
        """Puts a log message into the queue for the GUI and prints to console."""
        print(f"TRAINER LOG: {message}")
        self.update_queue.put({"type": "log", "message": message})
        
    def _put_error(self, message: str):
        """Puts an error message into the queue for the GUI and prints to console."""
        print(f"TRAINER ERROR: {message}")
        self.update_queue.put({"type": "error", "message": message})