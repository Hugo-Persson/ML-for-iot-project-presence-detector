#!/usr/bin/env python3
"""CLI app to interface with the Arduino ghost detector."""

import argparse
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Literal

import serial
import serial.tools.list_ports


DEFAULT_BAUD_RATE = 115200
COMMANDS = {
    "0": "Train as no_presence",
    "1": "Train as presence",
    "x": "Stop training (inference only)",
    "r": "Reset weights",
    "h": "Show help",
}


@dataclass
class PresenceReading:
    """A single presence reading from the detector."""

    prob: float
    ema: float
    presence: bool


@dataclass
class TrainingStatus:
    """Training status update from the detector."""

    label: int
    prob: float
    loss: float
    updates: int


def list_serial_ports() -> list[str]:
    """List available serial ports."""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def find_arduino_port() -> str | None:
    """Try to find an Arduino port automatically."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "usbmodem" in port.device.lower() or "arduino" in port.description.lower():
            return port.device
    return None


def parse_presence_line(line: str) -> tuple[Literal["prob", "ema", "presence"], float | bool] | None:
    """Parse a presence-related output line."""
    if line.startswith("Presence prob:"):
        try:
            value = float(line.split(":")[1].strip())
            return ("prob", value)
        except (ValueError, IndexError):
            return None
    elif line.startswith("Presence EMA:"):
        try:
            value = float(line.split(":")[1].strip())
            return ("ema", value)
        except (ValueError, IndexError):
            return None
    elif line.startswith("Presence:"):
        value = line.split(":")[1].strip()
        return ("presence", value == "YES")
    return None


def parse_training_line(line: str) -> TrainingStatus | None:
    """Parse a training status line."""
    match = re.match(r"Train y=(\d) p=([\d.]+) loss=([\d.]+) updates=(\d+)", line)
    if match:
        return TrainingStatus(
            label=int(match.group(1)),
            prob=float(match.group(2)),
            loss=float(match.group(3)),
            updates=int(match.group(4)),
        )
    return None


def format_presence_bar(value: float, width: int = 30) -> str:
    """Create a visual bar for presence probability."""
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def format_presence_output(prob: float, ema: float, presence: bool) -> str:
    """Format presence data for display."""
    bar = format_presence_bar(ema)
    presence_str = "\033[92mYES\033[0m" if presence else "\033[91mNO\033[0m"
    return f"[{bar}] {ema:.2%} | Raw: {prob:.2%} | {presence_str}"


def format_training_output(status: TrainingStatus) -> str:
    """Format training status for display."""
    label_str = "\033[92mpresence\033[0m" if status.label == 1 else "\033[91mno_presence\033[0m"
    return f"Training [{label_str}] prob={status.prob:.4f} loss={status.loss:.4f} updates={status.updates}"


class GhostDetectorCLI:
    """CLI interface for the ghost detector."""

    def __init__(self, port: str, baud_rate: int = DEFAULT_BAUD_RATE) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.serial: serial.Serial | None = None
        self.running = False
        self.read_thread: threading.Thread | None = None
        self._current_reading: dict[str, float | bool] = {}

    def connect(self) -> bool:
        """Connect to the serial port."""
        try:
            self.serial = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            time.sleep(2)  # Wait for Arduino reset
            return True
        except serial.SerialException as e:
            print(f"\033[91mError connecting to {self.port}: {e}\033[0m")
            return False

    def disconnect(self) -> None:
        """Disconnect from the serial port."""
        self.running = False
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)
        if self.serial and self.serial.is_open:
            self.serial.close()

    def send_command(self, cmd: str) -> None:
        """Send a single-character command to the Arduino."""
        if self.serial and self.serial.is_open:
            self.serial.write(cmd.encode())

    def _read_loop(self) -> None:
        """Background thread to read serial output."""
        while self.running and self.serial and self.serial.is_open:
            try:
                if self.serial.in_waiting > 0:
                    line = self.serial.readline().decode("utf-8", errors="replace").strip()
                    if line:
                        self._process_line(line)
            except serial.SerialException:
                break

    def _process_line(self, line: str) -> None:
        """Process a line of output from the Arduino."""
        # Check for training status
        training = parse_training_line(line)
        if training:
            print(f"\r{format_training_output(training)}")
            return

        # Check for presence data
        parsed = parse_presence_line(line)
        if parsed:
            key, value = parsed
            self._current_reading[key] = value

            # When we have all three values, display them
            if "prob" in self._current_reading and "ema" in self._current_reading and "presence" in self._current_reading:
                output = format_presence_output(
                    self._current_reading["prob"],
                    self._current_reading["ema"],
                    self._current_reading["presence"],
                )
                print(f"\r{output}")
                self._current_reading.clear()
            return

        # Check for status messages
        if "Training label set" in line or "Training paused" in line or "reset" in line.lower():
            print(f"\033[93m{line}\033[0m")
            return

        # Print other lines as-is (startup messages, help, etc.)
        if line and not line.startswith("Presence"):
            print(line)

    def start_reading(self) -> None:
        """Start the background reading thread."""
        self.running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()

    def run_interactive(self) -> None:
        """Run interactive mode."""
        print("\n\033[1mGhost Detector CLI\033[0m")
        print("=" * 40)
        print("Commands:")
        for cmd, desc in COMMANDS.items():
            print(f"  \033[1m{cmd}\033[0m - {desc}")
        print("  \033[1mq\033[0m - Quit")
        print("=" * 40)
        print()

        self.start_reading()

        try:
            while self.running:
                try:
                    user_input = input()
                    if user_input.lower() == "q":
                        break
                    elif user_input in COMMANDS:
                        self.send_command(user_input)
                    elif user_input:
                        print(f"\033[91mUnknown command: {user_input}\033[0m")
                        print("Type 'h' for help or 'q' to quit")
                except EOFError:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            print("\nDisconnecting...")
            self.disconnect()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interface with the Arduino ghost detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands (in interactive mode):
  0   Train as no_presence
  1   Train as presence
  x   Stop training (inference only)
  r   Reset weights
  h   Show help
  q   Quit

Examples:
  %(prog)s                     # Auto-detect port and run
  %(prog)s --port /dev/ttyUSB0 # Use specific port
  %(prog)s --list              # List available ports
  %(prog)s --send 1            # Send command and exit
""",
    )
    parser.add_argument(
        "--port",
        "-p",
        help="Serial port (auto-detected if not specified)",
    )
    parser.add_argument(
        "--baud",
        "-b",
        type=int,
        default=DEFAULT_BAUD_RATE,
        help=f"Baud rate (default: {DEFAULT_BAUD_RATE})",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available serial ports",
    )
    parser.add_argument(
        "--send",
        "-s",
        choices=["0", "1", "x", "r", "h"],
        help="Send a single command and exit",
    )
    parser.add_argument(
        "--monitor",
        "-m",
        action="store_true",
        help="Monitor output only (no interactive input)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.list:
        ports = list_serial_ports()
        if ports:
            print("Available serial ports:")
            for port in ports:
                print(f"  {port}")
        else:
            print("No serial ports found")
        sys.exit(0)

    # Determine port
    port = args.port
    if not port:
        port = find_arduino_port()
        if not port:
            print("\033[91mNo Arduino port found. Use --port to specify manually.\033[0m")
            print("Use --list to see available ports.")
            sys.exit(1)
        print(f"Auto-detected port: {port}")

    # Create CLI instance
    cli = GhostDetectorCLI(port, args.baud)

    if not cli.connect():
        sys.exit(1)

    print(f"Connected to {port}")

    # Send single command mode
    if args.send:
        cli.send_command(args.send)
        time.sleep(0.5)  # Wait for response
        # Read any available output
        if cli.serial and cli.serial.in_waiting > 0:
            while cli.serial.in_waiting > 0:
                line = cli.serial.readline().decode("utf-8", errors="replace").strip()
                if line:
                    print(line)
        cli.disconnect()
        sys.exit(0)

    # Monitor mode (read-only)
    if args.monitor:
        print("Monitoring output (Ctrl+C to exit)...")
        cli.start_reading()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            cli.disconnect()
        sys.exit(0)

    # Interactive mode
    cli.run_interactive()


if __name__ == "__main__":
    main()
