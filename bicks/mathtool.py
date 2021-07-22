import numpy as np

def dichotomy(f, a, b, epsilon=1.0e-5):
    """Tradional dichotomy to find a root of a function
    """
    fa = f(a)
    while True:
        c = (a + b) / 2.0
        if (b - a) <= epsilon:
            return c
        fc = f(c)
        if fc * fa < 0:
            b = c
        else:
            fa = fc
            a = c


def find_n_roots(f, n, deltax, eps=1.0e-10):
    """
    Warning! Don't make the deltax = 0.1

    """
    currentroot = 0
    root = []
    gox = 1.0e-5
    b = f(gox)
    while currentroot < n:
        a = b
        b = f(gox)
        if (a * b) < 0:
            root.append(dichotomy(f, gox - deltax, gox, epsilon=eps))
            currentroot = currentroot + 1
        gox = gox + deltax
    return root


def find_real_roots(f, endkz, startkz=0, deltakz=0.12, eps=1.0e-10):
    root = []
    kz = startkz
    b = f(kz)
    while kz < endkz:
        a = b
        b = f(kz)
        if (a * b) < 0:
            root.append(dichotomy(f, kz - deltakz, kz, epsilon=eps))
        kz = kz + deltakz
    return root

def find_proj_roots(f, endk0, startk0 = 0.121, deltak0 = 0.12, eps=1.0e-10):
    root = []
    k0 = startk0
    b = f(k0)
    while k0 < endk0:
        a = b
        b = f(k0)
        if (a * b) < 0:
            root.append(dichotomy(f, k0 - deltak0, k0, epsilon=eps))
        k0 = k0 + deltak0
    return root


def golden_section(f, a, b, epsilon=1.0e-10):
    c = a + 0.382 * (b - a)
    d = a + 0.618 * (b - a)
    while (b - a) > epsilon:
        if (abs(f(c))) < (abs(f(d))):
            a = c
            c = d
            d = a + 0.618 * (b - a)
        else:
            b = d
            d = c
            c = a + 0.382 * (b - a)
    return (b + a) / 2


def find_n_roots_for_small_and_big_q(f, qa, n, gox=0, deltax=0.024, eps=1.0e-10, peak1 = 0):
    currentroot = 0
    root = []

    @minus_cosqa(np.cos(qa))
    def nf(x):
        return f(x)
    while currentroot < n:
        if abs(nf(gox)) > 0.9:
            a = gox
            while abs(nf(gox)) > 0.9:
                gox = gox + deltax
            b = gox
            peak2 = golden_section(nf, a, b, epsilon=eps)
            if(f(peak1) * f(peak2)) <= eps:
                mayberoot = dichotomy(f, peak1, peak2, epsilon=eps)
                if abs(f(mayberoot)) <= eps:
                    root.append(mayberoot)
                    currentroot = currentroot + 1
            peak1 = peak2
        gox = gox + deltax
    return root


def find_real_roots_for_small_and_big_q(f, qa, deltax=0.024, eps=1.0e-10):
    root = []
    gox = 0
    peak1 = 0
    @minus_cosqa(np.cos(qa))
    def nf(x):
        return f(x)
    absnf = abs(nf(gox))
    while True: 
        if absnf > 0.9:
            a = gox
            while absnf > 0.9:
                gox = gox + deltax
                absnf = abs(nf(gox))
                if absnf > 1.0e3:
                    peak2 = gox    
                    if(f(peak1) * f(peak2)) <= 1.0e-8:
                        root.append(dichotomy(f, peak1, peak2, epsilon=eps))
                    return root
            b = gox
            peak2 = golden_section(nf, a, b, epsilon=eps)

            if(f(peak1) * f(peak2)) <= 1.0e-8:
                mayberoot = dichotomy(f, peak1, peak2, epsilon=eps)
                root.append(mayberoot)
                peak1 = peak2
        gox = gox + deltax
        absnf = abs(nf(gox))


def minus_cosqa(x):
    def minus(f):
        def wrapper(*args, **kargs):
            return f(*args, **kargs) - x
        return wrapper
    return minus


def find_all_peaks(f, x_start, x_end, deltax=0.01, eps=1.0e-3, lastdata=[]):
    x = x_start
    peaks = []
    if len(lastdata):
        a = 1
    while x < x_end:

        if f(x) > 0.8:
            a = x
            while f(x) > 0.8:
                x = x + deltax
                if x >= x_end:
                    break
            b = x
            peak = golden_section(f, a, b, epsilon=eps)
            fp = f(peak)
            if 0.99 < fp < 1.01:
                peaks.append(peak)
        x = x + deltax
    return peaks

def secant(f, a, b, eps=1.0e-5):
    fa = f(a)
    fb = f(b)
    c = b - ((b - a) / (fb - fa)) * fb
    while True:    
        if abs(b - a)<eps:
            return c
        else:
            c = b - ((b - a) / (fb - fa)) * f(b)
            a = b
            b = c
            fa = fb
            fb = f(b)