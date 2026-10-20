"""
data_sintetis.py
----------------
Membuat "alert deforestasi" sintetis berukuran 16x16 piksel (1 piksel ~ 10 m,
jadi 1 patch ~ 160 m x 160 m), meniru tumpukan citra:

  Sentinel-2 (optik) : B4 (Merah), B8 (NIR), B11 (SWIR)
  Sentinel-1 (radar) : VV, VH  (dinormalisasi 0-1, dengan speckle)

Setiap patch berisi SATU penyebab langsung (driver) deforestasi yang umum di
Sulawesi Tenggara. Bentuk spasial tiap driver berbeda-beda, sehingga CNN harus
belajar membaca POLA, bukan hanya nilai warna.

Tutupan awan hanya merusak kanal optik. Kanal radar tetap bersih
(radar menembus awan) -> inilah yang diuji pada studi kasus.
"""
import numpy as np

KELAS = [
    "Tambang Nikel Terbuka",
    "Perkebunan Skala Besar",
    "Ladang Skala Kecil",
    "Pembukaan Jalan",
    "Kebakaran Hutan",
]
KANAL = ["B4_Merah", "B8_NIR", "B11_SWIR", "S1_VV", "S1_VH"]
UKURAN = 16

# Nilai "spektral" rata-rata tiap tutupan: [Merah, NIR, SWIR, VV, VH]
SIGNATUR = {
    "hutan":      [0.04, 0.35, 0.15, 0.60, 0.55],
    "tambang":    [0.18, 0.22, 0.32, 0.45, 0.25],   # tanah laterit merah
    "kebun":      [0.10, 0.25, 0.25, 0.50, 0.38],
    "kebun_muda": [0.07, 0.31, 0.20, 0.52, 0.45],
    "ladang":     [0.09, 0.28, 0.24, 0.48, 0.36],
    "jalan":      [0.20, 0.20, 0.30, 0.40, 0.20],
    "terbakar":   [0.05, 0.12, 0.28, 0.50, 0.33],
}


def _grid():
    y, x = np.mgrid[0:UKURAN, 0:UKURAN]
    return y, x


def _mask_tambang(rng):
    """Bukaan besar tak beraturan + teras bertingkat (cincin konsentris)."""
    y, x = _grid()
    cy, cx = rng.uniform(5, 11, 2)
    ry, rx = rng.uniform(4, 7, 2)
    d = np.sqrt(((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2)
    mask = d < 1 + rng.normal(0, 0.08, (UKURAN, UKURAN))
    teras = (np.floor(d * 4) % 2 == 0) & mask
    return mask, teras


def _mask_kebun(rng):
    """Blok persegi besar dengan baris tanam yang teratur."""
    mask = np.zeros((UKURAN, UKURAN), bool)
    h, w = rng.integers(9, 15, 2)
    y0, x0 = rng.integers(0, UKURAN - h + 1), rng.integers(0, UKURAN - w + 1)
    mask[y0:y0 + h, x0:x0 + w] = True
    y, x = _grid()
    baris = ((x if rng.random() < 0.5 else y) % 3 == 0) & mask
    return mask, baris


def _mask_ladang(rng):
    """Beberapa petak kecil yang tersebar (pola mosaik)."""
    mask = np.zeros((UKURAN, UKURAN), bool)
    for _ in range(rng.integers(2, 5)):
        h, w = rng.integers(2, 5, 2)
        y0, x0 = rng.integers(0, UKURAN - h), rng.integers(0, UKURAN - w)
        mask[y0:y0 + h, x0:x0 + w] = True
    return mask, np.zeros_like(mask)


def _mask_jalan(rng):
    """Garis tipis memanjang, kadang bercabang."""
    y, x = _grid()
    mask = np.zeros((UKURAN, UKURAN), bool)
    for _ in range(1 + (rng.random() < 0.4)):
        sudut = rng.uniform(0, np.pi)
        cy, cx = rng.uniform(4, 12, 2)
        jarak = np.abs((x - cx) * np.sin(sudut) - (y - cy) * np.cos(sudut))
        mask |= jarak < rng.uniform(0.6, 1.2)
    return mask, np.zeros_like(mask)


def _mask_terbakar(rng):
    """Gabungan beberapa lingkaran (bentuk api menjalar) + sisa pohon."""
    y, x = _grid()
    mask = np.zeros((UKURAN, UKURAN), bool)
    for _ in range(rng.integers(3, 6)):
        cy, cx = rng.uniform(2, 14, 2)
        mask |= (y - cy) ** 2 + (x - cx) ** 2 < rng.uniform(4, 16)
    sisa = mask & (rng.random((UKURAN, UKURAN)) < 0.2)
    return mask & ~sisa, np.zeros_like(mask)


_PEMBUAT = [
    (_mask_tambang, "tambang", None),
    (_mask_kebun, "kebun", "kebun_muda"),
    (_mask_ladang, "ladang", None),
    (_mask_jalan, "jalan", None),
    (_mask_terbakar, "terbakar", None),
]


def _awan(rng, fraksi):
    """Masker awan halus: noise kasar 4x4 diperbesar lalu di-threshold."""
    if fraksi <= 0:
        return np.zeros((UKURAN, UKURAN), bool)
    if fraksi >= 1:
        return np.ones((UKURAN, UKURAN), bool)
    kasar = rng.random((4, 4))
    halus = np.kron(kasar, np.ones((4, 4))) + rng.normal(0, 0.15, (UKURAN, UKURAN))
    return halus > np.quantile(halus, 1 - fraksi)


def buat_patch(label, rng, fraksi_awan=0.0):
    """Hasilkan satu patch (5, 16, 16) untuk driver `label`."""
    fungsi, utama, sekunder = _PEMBUAT[label]
    mask, pola = fungsi(rng)

    img = np.empty((5, UKURAN, UKURAN))
    hutan = np.array(SIGNATUR["hutan"])[:, None, None]
    img[:] = hutan * rng.uniform(0.9, 1.1)          # variasi kondisi hutan
    img[:, mask] = np.array(SIGNATUR[utama])[:, None]
    if pola.any():
        if sekunder:
            img[:, pola] = np.array(SIGNATUR[sekunder])[:, None]
        else:  # teras tambang sedikit lebih terang di SWIR & lebih kasar di VV
            img[2, pola] += 0.05
            img[3, pola] += 0.06

    # Noise sensor optik + speckle radar.
    # Radar = komposit median ~5 akuisisi S1 dalam sebulan (setara ~20 look),
    # sehingga speckle lebih halus dibanding satu citra tunggal.
    img[:3] += rng.normal(0, 0.015, (3, UKURAN, UKURAN))
    img[3:] *= rng.gamma(20.0, 1 / 20.0, (2, UKURAN, UKURAN))

    # Awan: hanya merusak kanal optik
    awan = _awan(rng, fraksi_awan)
    img[:3, awan] = rng.uniform(0.55, 0.8) + rng.normal(0, 0.03, (3, awan.sum()))
    return np.clip(img, 0, 1).astype(np.float32)


def buat_dataset(n_per_kelas, seed=0, awan_min=0.0, awan_max=0.0):
    """Dataset seimbang. Fraksi awan diambil acak di [awan_min, awan_max]."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label in range(len(KELAS)):
        for _ in range(n_per_kelas):
            X.append(buat_patch(label, rng, rng.uniform(awan_min, awan_max)))
            y.append(label)
    X, y = np.stack(X), np.array(y)
    urut = rng.permutation(len(y))
    return X[urut], y[urut]
