#!/usr/bin/env python3.11
from larik import larik as lr
from larik import helpers
import numpy as np
import os,statistics,time,unittest,timeit



class TestLinAlg(unittest.TestCase):
  def setUp(self) -> None:
    N = 5084
    self.N = N
    self.la = lr.rand((N, N))
    self.lb = lr.rand((N, N))
    self.nla = self.la.numpy()
    self.nlb = self.lb.numpy()
    self.na = np.random.rand(N, N)
    self.nb = np.random.rand(N, N)

  def test_matmul(self):
    """Test matrix multiplication performance between larik and numpy implementations."""
    RUNS = 5  # Number of runs for accurate timing
    print(f"{helpers.Colors.HEADER}🚀 Matrix Multiplication Benchmark Started 🚀{helpers.Colors.END}")
    print(f"Running each matrix multiplication {RUNS} times for accuracy...\n")

    # Measure larik matrix multiplication
    larik_times = []
    for _ in range(RUNS):
      tic = time.monotonic()
      lc = self.la @ self.lb
      larik_times.append(time.monotonic() - tic)

    # Measure numpy matrix multiplication
    numpy_times = []
    for _ in range(RUNS):
      tic = time.monotonic()
      nc = self.na @ self.nb
      numpy_times.append(time.monotonic() - tic)

    # Perform additional numpy multiplication (nla @ nlb)
    nc = self.nla @ self.nlb

    # Verify matrix properties
    self.assertEqual(lc.size, nc.size, "Matrix sizes do not match!")
    self.assertEqual(lc.ndim, nc.ndim, "Matrix dimensions do not match!")
    self.assertTupleEqual(lc.shape, nc.shape, "Matrix shapes do not match!")

    # Calculate statistics
    avg_time_larik = statistics.mean(larik_times)
    min_time_larik = min(larik_times)
    max_time_larik = max(larik_times)
    avg_time_numpy = statistics.mean(numpy_times)
    min_time_numpy = min(numpy_times)
    max_time_numpy = max(numpy_times)

    # Display results if WITH_TIME is set
    if os.environ.get("WITH_TIME"):
      print(f"{helpers.Colors.BLUE}=== larik Performance ==={helpers.Colors.END}")
      print(
          f"Average Time: {helpers.colored(f'{avg_time_larik:.6f}', 'green')} seconds")
      print(f"Min Time: {helpers.colored(f'{min_time_larik:.6f}', 'yellow')} seconds")
      print(
          f"Max Time: {helpers.colored(f'{max_time_larik:.6f}', 'yellow')} seconds\n")

      print(f"{helpers.Colors.BLUE}=== numpy Performance ==={helpers.Colors.END}")
      print(f"Average Time: {helpers.colored(f'{avg_time_numpy:.6f}', 'red')} seconds")
      print(f"Min Time: {helpers.colored(f'{min_time_numpy:.6f}', 'red')} seconds")
      print(f"Max Time: {helpers.colored(f'{max_time_numpy:.6f}', 'red')} seconds\n")

      print(f"{helpers.Colors.BLUE}=== Matrix Outputs ==={helpers.Colors.END}")
      print(f"larik Result: {helpers.colored(str(lc.numpy()), 'green')}")
      print(f"numpy Result: {helpers.colored(str(nc), 'red')}\n")

      print(f"{helpers.Colors.HEADER}🏁 Performance Comparison 🏁{helpers.Colors.END}")
      helpers.get_ratio_time(avg_time_larik, avg_time_numpy)

  def test_add(self):
    """Test Add performance between larik and numpy implementations."""
    RUNS = 5  # Number of runs for accurate timing
    print(f"{helpers.Colors.HEADER}🚀 Matrix Multiplication Benchmark Started 🚀{helpers.Colors.END}")
    print(f"Running each add multiplication {RUNS} times for accuracy...\n")

    # Measure numpy matrix multiplication
    numpy_times = []
    for _ in range(RUNS):
      tic = time.monotonic()
      na = np.random.rand(self.N,self.N)
      nb = np.random.rand(self.N,self.N)
      nc = na + nb
      numpy_times.append(time.monotonic() - tic)

    # Measure larik matrix multiplication
    larik_times = []
    for _ in range(RUNS):
      tic = time.monotonic()
      la = lr.rand((self.N,self.N))
      lb = lr.rand((self.N,self.N))
      lc = la + lb
      larik_times.append(time.monotonic() - tic)


    # Calculate statistics
    avg_time_larik = statistics.mean(larik_times)
    min_time_larik = min(larik_times)
    max_time_larik = max(larik_times)
    avg_time_numpy = statistics.mean(numpy_times)
    min_time_numpy = min(numpy_times)
    max_time_numpy = max(numpy_times)

    print(f"{helpers.Colors.BLUE}=== larik Performance ==={helpers.Colors.END}")
    print(
        f"Average Time: {helpers.colored(f'{avg_time_larik:.6f}', 'green')} seconds")
    print(f"Min Time: {helpers.colored(f'{min_time_larik:.6f}', 'yellow')} seconds")
    print(
        f"Max Time: {helpers.colored(f'{max_time_larik:.6f}', 'yellow')} seconds\n")

    print(f"{helpers.Colors.BLUE}=== numpy Performance ==={helpers.Colors.END}")
    print(f"Average Time: {helpers.colored(f'{avg_time_numpy:.6f}', 'red')} seconds")
    print(f"Min Time: {helpers.colored(f'{min_time_numpy:.6f}', 'red')} seconds")
    print(f"Max Time: {helpers.colored(f'{max_time_numpy:.6f}', 'red')} seconds\n")

    helpers.get_ratio_time(avg_time_larik, avg_time_numpy)

if __name__ == "__main__":
  unittest.main()
