"""Score + SFX for the SOVOPT brag video, synthesised as one piece.

A minor, 100 BPM (beat 0.6 s, bar 2.4 s); scene cuts sit on the bar grid.
Effects are pitched into the key and share the music's reverb.
"""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR, DUR = 48000, 20.0
N = int(SR * DUR)
BEAT, BAR = 0.6, 2.4
rng = np.random.default_rng(7)

def hz(m): return 440.0 * 2 ** ((m - 69) / 12)
def db(x): return 10 ** (x / 20)
def bp(x, lo, hi, o=2): return sosfilt(butter(o, [lo, hi], "band", fs=SR, output="sos"), x)
def hp(x, f, o=2): return sosfilt(butter(o, f, "high", fs=SR, output="sos"), x)
def lp(x, f, o=2): return sosfilt(butter(o, f, "low", fs=SR, output="sos"), x)

music = np.zeros((N, 2)); sfx = np.zeros((N, 2))
send_m = np.zeros((N, 2)); send_s = np.zeros((N, 2))

def put(buf, t0, sig, gain=1.0, pan=0.0, send=None, send_amt=0.0):
    i = int(t0 * SR)
    if i >= N: return
    sig = sig[: N - i] * gain
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    buf[i:i + len(sig), 0] += sig * l * 1.414
    buf[i:i + len(sig), 1] += sig * r * 1.414
    if send is not None:
        send[i:i + len(sig), 0] += sig * l * send_amt * 1.414
        send[i:i + len(sig), 1] += sig * r * send_amt * 1.414

def tt(d): return np.arange(int(d * SR)) / SR

# ---------- instruments ----------
def pad(m, d, att=0.6, rel=1.0):
    t = tt(d + rel); f = hz(m); s = np.zeros_like(t)
    for det in (-0.0035, 0.0, 0.0035):
        for n in range(1, 7):
            s += np.sin(2 * np.pi * f * (1 + det) * n * t + n * 1.3) / n ** 1.6
    env = np.minimum(1, t / att) * np.where(t < d, 1, np.exp(-(t - d) / (rel / 3)))
    return lp(s * env / 3, 2200)

def pluck(m, d=1.2, bright=1.0):
    t = tt(d); f = hz(m); s = np.zeros_like(t)
    for n in range(1, 7):
        s += np.sin(2 * np.pi * f * n * t) * np.exp(-t * (5 + 3 * n / bright)) / n
    return s * np.minimum(1, t / 0.003)

def bass(m, d=0.3):
    t = tt(d + 0.05); f = hz(m)
    s = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t)
    env = np.minimum(1, t / 0.006) * np.exp(-t * 5) * np.where(t < d, 1, np.exp(-(t - d) / 0.015))
    return np.tanh(1.4 * s * env) / 1.4

def kick():
    t = tt(0.45); ph = 2 * np.pi * np.cumsum(46 + 90 * np.exp(-t * 32)) / SR
    return np.sin(ph) * np.exp(-t * 8) + 0.15 * lp(rng.standard_normal(len(t)), 3000) * np.exp(-t * 120)

def hat(d=0.08):
    t = tt(d); return hp(rng.standard_normal(len(t)), 7500) * np.exp(-t * 70)

def clap():
    t = tt(0.25); return bp(rng.standard_normal(len(t)), 900, 3200) * np.exp(-t * 22)

def bell(m, d=2.2):
    t = tt(d); f = hz(m); s = np.zeros_like(t)
    for r, a, k in ((1, 1, 2.4), (2.0, .45, 3.5), (3.0, .22, 5), (4.16, .12, 7)):
        s += a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * k)
    return s * np.minimum(1, t / 0.002)

def tick():
    t = tt(0.012); return bp(rng.standard_normal(len(t)), 2500, 6000) * np.exp(-t * 500)

def swish(d=0.35, lo=600, hi=5000):
    t = tt(d); n = bp(rng.standard_normal(len(t)), lo, hi)
    return n * np.sin(np.pi * t / d) ** 2

def riser(d):
    t = tt(d); n = hp(rng.standard_normal(len(t)), 1500) * (t / d) ** 3
    g = np.sin(2 * np.pi * np.cumsum(hz(57) * 2 ** (2 * t / d)) / SR) * (t / d) ** 2
    return 0.6 * n + 0.25 * g

def impact():
    t = tt(1.6)
    return np.sin(2 * np.pi * 52 * t) * np.exp(-t * 2.6) + 0.3 * lp(rng.standard_normal(len(t)), 400) * np.exp(-t * 10)

# ---------- harmony ----------
A, C, D, E, F, G = 57, 60, 62, 64, 65, 67
CH = [(0.0, 3.6, [A, C, E], A - 24), (3.6, 4.8, [C, E, G], C - 24), (4.8, 7.2, [F - 12, A, C, E], F - 24),
      (7.2, 9.6, [A, C, E], A - 24), (9.6, 12.0, [F - 12, A, C], F - 24), (12.0, 14.4, [C, E, G], C - 24),
      (14.4, 16.8, [G - 12, B := 59, D], G - 24), (16.8, 18.0, [F - 12, A, C, E], F - 24),
      (18.0, 19.2, [G - 12, B, D], G - 24), (19.2, 20.0, [C, E, G, 72], C - 24)]

def chord_at(t):
    for a, b, notes, root in CH:
        if a <= t < b: return notes, root
    return CH[-1][2], CH[-1][3]

# pad (first section quieter; blooms at the reveal)
for a, b, notes, root in CH:
    g = db(-25) if a < 3.6 else db(-21)
    rel = 1.6 if a >= 19.2 else 0.9
    for m in notes:
        put(music, a, pad(m, b - a, att=0.9 if a == 0 else 0.35, rel=rel), g, pan=(m % 5 - 2) * 0.12, send=send_m, send_amt=0.5)

kicks = []
# hook: soft pulsing bass on eighths
for k in range(12):
    t0 = k * BEAT / 2
    put(music, t0, bass(A - 24, 0.22), db(-20) * (0.5 + 0.5 * k / 12))
# groove 7.2 -> 16.8
t0 = 7.2
while t0 < 16.8 - 1e-6:
    k = round((t0 - 7.2) / (BEAT / 2))
    notes, root = chord_at(t0)
    if k % 2 == 0:
        put(music, t0, kick(), db(-11)); kicks.append(t0)
    else:
        put(music, t0, hat(), db(-27), pan=0.35)
    if k % 4 == 2 and t0 >= 12.0:
        put(music, t0, clap(), db(-26), send=send_m, send_amt=0.6)
    put(music, t0, bass(root + (12 if k % 4 == 3 else 0)), db(-15))
    arp = sorted(notes)
    m = arp[k % len(arp)] + (12 if (k // len(arp)) % 2 else 0)
    put(music, t0, pluck(m + 12, 0.9), db(-25), pan=-0.4 if k % 2 else 0.4, send=send_m, send_amt=0.45)
    t0 += BEAT / 2
# reveal: bass + slow arp, no drums
for k in range(12):
    t0 = 3.6 + k * BEAT / 2
    notes, root = chord_at(t0)
    if k % 2 == 0: put(music, t0, bass(root, 0.5), db(-17))
    m = sorted(notes)[k % len(notes)] + 12
    put(music, t0, pluck(m, 1.2, 0.6), db(-29), pan=-0.3 if k % 2 else 0.3, send=send_m, send_amt=0.6)
# outro: half-time kicks, arp, final chord
for k in range(8):
    t0 = 16.8 + k * BEAT / 2
    notes, root = chord_at(t0)
    if k % 4 == 0: put(music, t0, kick(), db(-13)); kicks.append(t0)
    if k % 2 == 0: put(music, t0, bass(root, 0.5), db(-16))
    put(music, t0, pluck(sorted(notes)[k % len(notes)] + 12, 1.0), db(-26), pan=-0.35 if k % 2 else 0.35, send=send_m, send_amt=0.5)
put(music, 19.2, kick(), db(-13)); kicks.append(19.2)
put(music, 19.2, bass(C - 24, 0.8), db(-15))
for m in (72, 76, 79):
    put(music, 19.2, bell(m, 0.8), db(-27), send=send_m, send_amt=0.7)

# ---------- SFX, in key ----------
# hook: each strike-through = a soft swish + a pentatonic pluck
for i, m in enumerate([69, 72, 74, 76, 79, 81]):
    a = 0.12 + i * 0.2 + 0.2
    put(sfx, a, swish(0.18, 1200, 5000), db(-24), pan=-0.5 + i * 0.2, send=send_s, send_amt=0.3)
    put(sfx, a, pluck(m, 0.8, 1.4), db(-22), pan=-0.5 + i * 0.2, send=send_s, send_amt=0.4)
put(sfx, 1.55, pluck(45, 1.5, 0.5), db(-18), send=send_s, send_amt=0.3)          # "No solver library."
put(sfx, 3.6 - 1.0, riser(1.0), db(-24), send=send_s, send_amt=0.4)
put(sfx, 3.6, impact(), db(-12), send=send_s, send_amt=0.25)
put(sfx, 6.3, bell(84, 1.6), db(-30), pan=0.5, send=send_s, send_amt=0.6)          # optimum vertex
# scene-cut whooshes
for c in (7.2, 12.0, 16.8):
    put(sfx, c - 0.3, swish(0.45, 400, 3500), db(-27), send=send_s, send_amt=0.5)
# terminal typing
CMD_LEN = 56
for j in range(CMD_LEN):
    tj = 7.75 + 0.8 * j / CMD_LEN + rng.uniform(-0.004, 0.004)
    put(sfx, tj, tick(), db(-33) * rng.uniform(0.6, 1.0), pan=rng.uniform(-0.2, 0.2))
# output lines: tiny blips on the A pentatonic
for tl, m in zip([8.68, 8.8, 8.92, 9.1, 9.18, 9.26, 9.34, 9.42, 9.5, 9.58],
                 [81, 84, 86, 81, 84, 86, 88, 91, 93, 96]):
    put(sfx, tl, pluck(m, 0.3, 2.0), db(-33), pan=0.2, send=send_s, send_amt=0.3)
# ACCEPTED
put(sfx, 10.25, bell(76, 2.0), db(-22), send=send_s, send_amt=0.5)
put(sfx, 10.37, bell(81, 2.0), db(-22), send=send_s, send_amt=0.5)
# engine cards land: C E G
for i, m in enumerate([72, 76, 79]):
    put(sfx, 12.4 + i * 0.18 + 0.1, pluck(m, 1.0, 1.2), db(-23), pan=-0.4 + 0.4 * i, send=send_s, send_amt=0.45)
put(sfx, 13.9, bell(88, 1.2), db(-31), send=send_s, send_amt=0.5)                 # HiGHS chip
for a, m in ((16.9, 77), (17.15, 81)):
    put(sfx, a, pluck(m, 1.0, 1.0), db(-25), send=send_s, send_amt=0.5)

# ---------- mix ----------
# sidechain the music gently to the kicks
duck = np.ones(N)
tN = np.arange(N) / SR
for k in kicks:
    i = int(k * SR); seg = tN[i:i + int(0.3 * SR)] - k
    duck[i:i + len(seg)] = np.minimum(duck[i:i + len(seg)], 1 - 0.28 * np.exp(-seg / 0.09))
music *= duck[:, None]

ir_t = tt(1.8)
ir = np.stack([lp(rng.standard_normal(len(ir_t)), 5000) * np.exp(-ir_t * 3.2) for _ in range(2)], 1)
ir /= np.abs(ir).sum(0).max() / 6
wet = np.stack([fftconvolve(send_m[:, c] + send_s[:, c], ir[:, c])[:N] for c in range(2)], 1)
wet = hp(wet, 180)

mix = music + sfx + db(-10) * wet
mix = hp(mix, 30)
# glue: gentle soft clip, fade the tail, normalise to -1 dBFS
mix = np.tanh(mix / (np.abs(mix).max() * 0.7))
fade = np.clip((DUR - tN) / 0.6, 0, 1)[:, None]
mix *= fade
mix *= db(-1) / np.abs(mix).max()
wavfile.write("audio.wav", SR, (mix * 32767).astype(np.int16))
print("peak", np.abs(mix).max(), "rms dBFS", 20 * np.log10(np.sqrt((mix ** 2).mean())))
