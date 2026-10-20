"""
main.py — Studi Kasus: Atribusi Driver Deforestasi di Sulawesi Tenggara
dengan Fusi Sentinel-1 (radar) + Sentinel-2 (optik) dan CNN (NumPy).

Jalankan:  python main.py
Hasil tersimpan di folder hasil/
"""
import csv
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_sintetis import KELAS, buat_dataset, buat_patch  # noqa: E402
from cnn_numpy import CNNDriver  # noqa: E402

HASIL = os.path.join(os.path.dirname(__file__), "hasil")
os.makedirs(HASIL, exist_ok=True)

SKENARIO = {
    "Optik saja (S2)": [0, 1, 2],
    "Fusi Optik + Radar (S2+S1)": [0, 1, 2, 3, 4],
}
TINGKAT_AWAN = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
EPOCH, BATCH = 25, 32


def latih(X, y, kanal, nama):
    model = CNNDriver(n_kanal=len(kanal), seed=42)
    rng = np.random.default_rng(7)
    Xk = X[:, kanal]
    print(f"\n[LATIH] {nama}")
    riwayat = []
    for ep in range(1, EPOCH + 1):
        urut = rng.permutation(len(y))
        lr = 0.003 if ep <= 15 else 0.001
        loss = np.mean([model.train_step(Xk[urut[i:i + BATCH]], y[urut[i:i + BATCH]], lr)
                        for i in range(0, len(y), BATCH)])
        riwayat.append(loss)
        print(f"  epoch {ep:2d}/{EPOCH}  loss = {loss:.4f}")
    return model, riwayat


def norm(X):
    return ((X - RATA) / SIMPANG).astype(np.float32)


def matriks_konfusi(y, pred, k=len(KELAS)):
    cm = np.zeros((k, k), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    return cm


def f1_per_kelas(cm):
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    return 2 * prec * rec / np.maximum(prec + rec, 1e-9)


def main():
    t0 = time.time()
    # Data latih: awan acak 0-100% (musim hujan tropis bisa tertutup total)
    X_tr, y_tr = buat_dataset(600, seed=1, awan_min=0.0, awan_max=1.0)
    print(f"Data latih: {X_tr.shape} | {len(KELAS)} kelas driver")

    # Standarisasi per kanal (mean/std dari data latih) agar skala optik & radar setara
    global RATA, SIMPANG
    RATA = X_tr.mean(axis=(0, 2, 3), keepdims=True)
    SIMPANG = X_tr.std(axis=(0, 2, 3), keepdims=True)
    X_tr = norm(X_tr)
    np.savez(os.path.join(HASIL, "normalisasi.npz"), rata=RATA, simpang=SIMPANG)

    model, loss = {}, {}
    for nama, kanal in SKENARIO.items():
        model[nama], loss[nama] = latih(X_tr, y_tr, kanal, nama)
        model[nama].simpan(os.path.join(HASIL, f"bobot_{'fusi' if len(kanal) == 5 else 'optik'}.npz"))

    # ---------------- Uji ketahanan terhadap awan
    print("\n[UJI] Akurasi pada berbagai tingkat tutupan awan")
    akurasi = {n: [] for n in SKENARIO}
    for i, awan in enumerate(TINGKAT_AWAN):
        X_te, y_te = buat_dataset(120, seed=100 + i, awan_min=awan, awan_max=awan)
        X_te = norm(X_te)
        baris = f"  awan {int(awan * 100):3d}% |"
        for nama, kanal in SKENARIO.items():
            pred = model[nama].predict_proba(X_te[:, kanal]).argmax(1)
            akurasi[nama].append((pred == y_te).mean())
            baris += f" {nama}: {akurasi[nama][-1] * 100:5.1f}% |"
        print(baris)

    # ---------------- Evaluasi detail pada awan 75%
    X_te, y_te = buat_dataset(120, seed=999, awan_min=0.75, awan_max=0.75)
    X_te = norm(X_te)
    cms, f1s = {}, {}
    for nama, kanal in SKENARIO.items():
        pred = model[nama].predict_proba(X_te[:, kanal]).argmax(1)
        cms[nama] = matriks_konfusi(y_te, pred)
        f1s[nama] = f1_per_kelas(cms[nama])

    # ---------------- Simpan metrik ke CSV
    with open(os.path.join(HASIL, "akurasi_vs_awan.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tutupan_awan_persen"] + list(SKENARIO))
        for i, awan in enumerate(TINGKAT_AWAN):
            w.writerow([int(awan * 100)] + [f"{akurasi[n][i]:.4f}" for n in SKENARIO])
    with open(os.path.join(HASIL, "f1_per_kelas_awan75.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["driver"] + list(SKENARIO))
        for k, kelas in enumerate(KELAS):
            w.writerow([kelas] + [f"{f1s[n][k]:.4f}" for n in SKENARIO])

    # ---------------- Simulasi laporan alert bulanan
    rng = np.random.default_rng(2026)
    n_alert = 40
    benar = rng.integers(0, len(KELAS), n_alert)
    awan = rng.uniform(0.0, 1.0, n_alert)
    X_al = norm(np.stack([buat_patch(c, rng, a) for c, a in zip(benar, awan)]))
    p = model["Fusi Optik + Radar (S2+S1)"].predict_proba(X_al)
    with open(os.path.join(HASIL, "laporan_alert_bulanan.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id_alert", "lintang", "bujur", "tutupan_awan_persen",
                    "driver_prediksi", "keyakinan", "driver_sebenarnya", "status"])
        for i in range(n_alert):
            lat, lon = rng.uniform(-4.30, -3.90), rng.uniform(121.65, 121.95)
            kp = p[i].argmax()
            status = "PERLU VERIFIKASI LAPANGAN" if p[i, kp] < 0.7 else "OTOMATIS"
            w.writerow([f"ALR-{i + 1:03d}", f"{lat:.4f}", f"{lon:.4f}", int(awan[i] * 100),
                        KELAS[kp], f"{p[i, kp]:.2f}", KELAS[benar[i]], status])

    buat_grafik(akurasi, cms, loss, f1s)
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik. Lihat folder hasil/")


# ====================================================================== grafik
WARNA = {"Optik saja (S2)": "#d97706", "Fusi Optik + Radar (S2+S1)": "#047857"}


def buat_grafik(akurasi, cms, loss, f1s):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # 1. Akurasi vs awan
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = [int(a * 100) for a in TINGKAT_AWAN]
    for nama, nilai in akurasi.items():
        ax.plot(x, np.array(nilai) * 100, "o-", lw=2.2, color=WARNA[nama], label=nama)
        geser = 8 if "Fusi" in nama else -14   # label fusi di atas, optik di bawah
        for xi, v in zip(x, nilai):
            ax.annotate(f"{v * 100:.0f}%", (xi, v * 100), textcoords="offset points",
                        xytext=(0, geser), ha="center", fontsize=8, color=WARNA[nama])
    ax.axhline(20, ls=":", color="gray", lw=1)
    ax.text(60, 22, "tebakan acak (20%)", ha="center", fontsize=8, color="gray")
    ax.set(xlabel="Tutupan awan pada citra optik (%)", ylabel="Akurasi (%)",
           ylim=(0, 108), title="Akurasi atribusi driver vs tutupan awan")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(HASIL, "1_akurasi_vs_awan.png"), dpi=150)
    plt.close(fig)

    # 2. Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    singkat = ["Tambang", "Kebun\nBesar", "Ladang\nKecil", "Jalan", "Kebakaran"]
    for ax, (nama, cm) in zip(axes, cms.items()):
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Greens" if "Fusi" in nama else "Oranges", vmin=0, vmax=1)
        for i in range(len(KELAS)):
            for j in range(len(KELAS)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cmn[i, j] > 0.55 else "black", fontsize=9)
        ax.set_xticks(range(5), singkat, fontsize=8)
        ax.set_yticks(range(5), singkat, fontsize=8)
        ax.set(xlabel="Prediksi", ylabel="Sebenarnya",
               title=f"{nama}\nakurasi {np.trace(cm) / cm.sum() * 100:.1f}% (awan 75%)")
    fig.tight_layout()
    fig.savefig(os.path.join(HASIL, "2_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # 3. Contoh patch tiap driver
    fig, axes = plt.subplots(3, 5, figsize=(11, 7))
    for k in range(5):
        bersih = buat_patch(k, np.random.default_rng(50 + k), 0.0)
        berawan = buat_patch(k, np.random.default_rng(50 + k), 0.5)
        rgb = lambda im: np.clip(im[[2, 1, 0]].transpose(1, 2, 0) * 2.2, 0, 1)  # SWIR-NIR-Merah
        axes[0, k].imshow(rgb(bersih))
        axes[1, k].imshow(rgb(berawan))
        axes[2, k].imshow(berawan[4], cmap="gray", vmin=0.15, vmax=0.7)
        axes[0, k].set_title(KELAS[k], fontsize=9)
    for r, lbl in enumerate(["S2 cerah\n(SWIR-NIR-R)", "S2 awan 50%", "S1 VH radar\n(menembus awan)"]):
        axes[r, 0].set_ylabel(lbl, fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Contoh alert deforestasi sintetis (16x16 piksel ≈ 160 m x 160 m)")
    fig.tight_layout()
    fig.savefig(os.path.join(HASIL, "3_contoh_patch.png"), dpi=150)
    plt.close(fig)

    # 4. Kurva loss + F1 per kelas
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for nama, l in loss.items():
        a1.plot(range(1, len(l) + 1), l, "o-", color=WARNA[nama], label=nama)
    a1.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Kurva pelatihan")
    a1.legend(frameon=False)
    lebar = 0.38
    idx = np.arange(5)
    for i, (nama, f1) in enumerate(f1s.items()):
        a2.bar(idx + (i - 0.5) * lebar, f1, lebar, color=WARNA[nama], label=nama)
    a2.set_xticks(idx, singkat, fontsize=8)
    a2.set(ylabel="F1-score", ylim=(0, 1.05), title="F1 per driver (awan 75%)")
    fig.tight_layout()
    fig.savefig(os.path.join(HASIL, "4_loss_dan_f1.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
