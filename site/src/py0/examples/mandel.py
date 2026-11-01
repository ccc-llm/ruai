# The Mandelbrot set in ASCII art.
chars = " .:-=+*#%@"
for row in range(24):
    y = -1.2 + row * 0.1
    line = ""
    for col in range(72):
        x = -2.1 + col * 0.04
        zx = 0.0
        zy = 0.0
        i = 0
        while i < 60 and zx * zx + zy * zy < 4.0:
            zx, zy = zx * zx - zy * zy + x, 2.0 * zx * zy + y
            i += 1
        line += chars[i * (len(chars) - 1) // 60]
    print(line)
