from typing import Optional

def colored(st, color:Optional[str], background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st  # replace the termcolor library with one line  # noqa: E501
class Colors:
  HEADER = '\033[95m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  RED = '\033[91m'
  YELLOW = '\033[93m'
  END = '\033[0m'

def get_ratio_time(time_larik, time_numpy):
  """Calculate and display the relative speed of larik vs numpy."""
  if time_numpy > time_larik and time_larik > 0:
    ratio = time_numpy / time_larik
    print(f"{Colors.GREEN}larik is {ratio:.2f} times faster than numpy!{Colors.END}")
  elif time_larik > time_numpy and time_numpy > 0:
    ratio = time_larik / time_numpy
    print(f"{Colors.RED}numpy is {ratio:.2f} times faster than larik!{Colors.END}")
  else: print(f"{Colors.YELLOW}Both implementations have identical execution times or one of the times is zero.{Colors.END}")
