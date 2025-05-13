import subprocess, sys, os, csv

# methods and grid sizes you want to test
methods = ["vector","numba","numba-parallel","mp"]
grids   = [1,2,3,4,5,6,7,8]
repeats = [1, 5]

# open CSV for writing
with open("cold_times.csv","w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["method","resolution", "repeats","time_s"])

    for r in repeats:
        for g in grids:
            for m in methods:
                cmd = [
                  sys.executable, "jpeg_compression.py",
                  "input.png",
                  f"--method={m}",
                  f"--grid={g}",
                  f"--repeat={r}",
                  "--quality=50"
                ]
                print("Running:", " ".join(cmd))
                # run it and capture the stdout line: e.g. "numba,1024x1024,1,0.1234"
                res = subprocess.run(cmd,
                                    capture_output=True,
                                    text=True,
                                    check=True)
                out_line = res.stdout.strip()
                print("  ->", out_line)
                # parse CSV‐style stdout and write to our file
                method, resolution, t = out_line.split(",")
                writer.writerow([method, resolution, int(r), float(t)])

print("All timings saved to cold_times.csv")
