import argparse
import queue
import threading
import time
import sys

# GUI imports - only if not running in headless mode
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Internal project imports
from trainer import TrainerThread

class TrainingApp:
    """
    The main application class for the DDPM training GUI.
    It sets up the Tkinter window, manages widgets, and orchestrates the
    training process in a background thread.
    """
    def __init__(self, root: tk.Tk):
        """
        Initializes the application, sets up the GUI, and prepares for training.

        Args:
            root (tk.Tk): The main Tkinter window.
        """
        self.root = root
        self.root.title("Text Diffusion Model Training")
        self.root.geometry("1000x800")

        # Communication queue for the trainer thread
        self.update_queue = queue.Queue()
        self.trainer_thread: threading.Thread | None = None

        # Data for plotting
        self.plot_steps = []
        self.plot_losses = []

        self.setup_ui()

        # Handle window closing gracefully
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start the training process shortly after the GUI is up
        self.root.after(100, self.start_training)

    def setup_ui(self):
        """
        Creates and arranges all the GUI widgets.
        """
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top frame for stats ---
        stats_frame = ttk.LabelFrame(main_frame, text="Live Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=5)

        self.epoch_var = tk.StringVar(value="Epoch: -")
        self.step_var = tk.StringVar(value="Step: -")
        self.loss_var = tk.StringVar(value="Loss: -")
        self.status_var = tk.StringVar(value="Status: Initializing...")

        ttk.Label(stats_frame, textvariable=self.epoch_var, font=("Helvetica", 12)).pack(side=tk.LEFT, padx=10)
        ttk.Label(stats_frame, textvariable=self.step_var, font=("Helvetica", 12)).pack(side=tk.LEFT, padx=10)
        ttk.Label(stats_frame, textvariable=self.loss_var, font=("Helvetica", 12)).pack(side=tk.LEFT, padx=10)
        
        status_label = ttk.Label(stats_frame, textvariable=self.status_var, font=("Helvetica", 12, "italic"))
        status_label.pack(side=tk.RIGHT, padx=10)
        stats_frame.pack_propagate(False) # Prevent resizing

        # --- Middle frame for plot and generated text ---
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10,0))
        content_frame.columnconfigure(0, weight=2) # Plot gets more space
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)

        # --- Plot Section ---
        plot_frame = ttk.LabelFrame(content_frame, text="Training Loss", padding="10")
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.fig = Figure(figsize=(5, 4), dpi=100, tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Loss vs. Steps")
        self.ax.set_xlabel("Training Steps")
        self.ax.set_ylabel("Loss (MSE)")
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Generated Text Section ---
        text_frame = ttk.LabelFrame(content_frame, text="Generated Sample", padding="10")
        text_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self.text_widget = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Courier", 11))
        self.text_widget.pack(fill=tk.BOTH, expand=True)

    def start_training(self):
        """
        Initializes and starts the training thread.
        """
        try:
            self.trainer_thread = TrainerThread(self.update_queue)
            self.trainer_thread.start()
            self.check_queue()  # Start polling the queue for updates
        except Exception as e:
            error_message = f"Failed to start training thread:\n{e}"
            self.status_var.set("Status: Error!")
            messagebox.showerror("Initialization Error", error_message)

    def check_queue(self):
        """
        Periodically checks the queue for data from the training thread
        and updates the GUI accordingly.
        """
        try:
            while not self.update_queue.empty():
                update = self.update_queue.get_nowait()
                msg_type = update.get("type")

                if msg_type == "stats":
                    self.epoch_var.set(f"Epoch: {update['epoch']}")
                    self.step_var.set(f"Step: {update['step']}")
                    loss = update['loss']
                    self.loss_var.set(f"Loss: {loss:.4f}")
                    
                    # Update plot data
                    self.plot_steps.append(update['step'])
                    self.plot_losses.append(loss)
                    self.update_plot()
                
                elif msg_type == "sample":
                    self.update_generated_text(update['text'])

                elif msg_type == "log":
                    self.status_var.set(f"Status: {update['message']}")

                elif msg_type == "error":
                    self.status_var.set("Status: Error!")
                    messagebox.showerror("Training Error", update['message'])
                    self.on_closing()

                elif msg_type == "done":
                    self.status_var.set("Status: Training Finished!")

        finally:
            # Schedule the next check
            self.root.after(100, self.check_queue)
            
    def update_plot(self):
        """
        Redraws the Matplotlib plot with the latest loss data.
        """
        self.ax.clear()
        if self.plot_steps:
            self.ax.plot(self.plot_steps, self.plot_losses, '-')
            # Auto-scale y-axis with some padding for better visibility
            min_loss = min(self.plot_losses)
            max_loss = max(self.plot_losses)
            padding = (max_loss - min_loss) * 0.1
            self.ax.set_ylim(bottom=max(0, min_loss - padding), top=max_loss + padding)

        self.ax.set_title("Loss vs. Steps")
        self.ax.set_xlabel("Training Steps")
        self.ax.set_ylabel("Loss (MSE)")
        self.ax.grid(True)
        self.canvas.draw()

    def update_generated_text(self, text: str):
        """
        Updates the text widget with a new sample sentence.

        Args:
            text (str): The new sentence to display.
        """
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete('1.0', tk.END)
        self.text_widget.insert(tk.END, text)
        self.text_widget.config(state=tk.DISABLED)

    def on_closing(self):
        """
        Handles the window close event to stop the training thread gracefully.
        """
        if self.trainer_thread and self.trainer_thread.is_alive():
            print("Stopping training thread...")
            self.status_var.set("Status: Stopping...")
            self.trainer_thread.stop()
            # Give the thread a moment to terminate before closing
            self.trainer_thread.join(timeout=2)
        
        print("Closing application.")
        self.root.destroy()

def run_console_training():
    """
    Runs the training process in the console without a GUI.
    """
    print("Running in command-line mode.")
    update_queue = queue.Queue()
    
    try:
        trainer_thread = TrainerThread(update_queue)
        trainer_thread.start()

        while trainer_thread.is_alive():
            try:
                update = update_queue.get(timeout=1.0) # Wait for 1s
                msg_type = update.get("type")

                if msg_type == "stats":
                    print(f"Epoch: {update['epoch']}, Step: {update['step']}, Loss: {update['loss']:.4f}")
                elif msg_type == "sample":
                    print(f"--- Generated Sample ---")
                    print(update['text'])
                    print("------------------------")
                elif msg_type == "log":
                    print(f"[LOG] {update['message']}")
                elif msg_type == "error":
                    print(f"[ERROR] {update['message']}", file=sys.stderr)
                    break # Exit on error
                elif msg_type == "done":
                    print("[INFO] Training finished.")
                    break

            except queue.Empty:
                # This is expected if the thread is working and hasn't sent an update
                pass

        # Wait for the thread to finish completely
        trainer_thread.join(timeout=5)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping training thread...")
        if trainer_thread and trainer_thread.is_alive():
            trainer_thread.stop()
            trainer_thread.join(timeout=5)
        print("Trainer stopped.")
    except Exception as e:
        import traceback
        print(f"A critical error occurred: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Text Diffusion Model.")
    parser.add_argument("--no-gui", action="store_true", help="Run training in the console without a GUI.")
    args = parser.parse_args()

    if args.no_gui:
        run_console_training()
    else:
        if not GUI_AVAILABLE:
            print("GUI libraries (tkinter, matplotlib) are not installed, but are required.")
            print("Please install them (e.g., 'pip install matplotlib') or run with --no-gui.")
            exit(1)
        try:
            root = tk.Tk()
            app = TrainingApp(root)
            root.mainloop()
        except Exception as e:
            import traceback
            # Fallback for critical errors during Tkinter setup
            print("A critical error occurred. Application will exit.")
            print(f"Error: {e}")
            print(traceback.format_exc())