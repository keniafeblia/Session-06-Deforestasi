# 📤 Panduan Upload ke GitHub

Ada dua cara. Pilih yang paling mudah untuk kamu.

---

## Cara A — Lewat Website GitHub (tanpa install apa pun)

1. Ekstrak file `deforestasi-driver-sultra.zip` di komputer.
2. Masuk ke [github.com](https://github.com) → klik **New** (tombol hijau) untuk membuat repository baru.
3. Isi:
   - **Repository name:** `deforestasi-driver-sultra`
   - **Description:** `Atribusi penyebab deforestasi di bawah awan dengan fusi Sentinel-1 + Sentinel-2 dan CNN NumPy`
   - Pilih **Public**
   - **Jangan** centang "Add a README file" (kita sudah punya)
4. Klik **Create repository**.
5. Di halaman repository kosong, klik link **uploading an existing file**.
6. Buka folder hasil ekstrak, **pilih semua isinya** (`main.py`, `README.md`, folder `src`, `hasil`, `docs`, dll.), lalu *drag & drop* ke halaman GitHub.
   > Tarik **isi** foldernya, bukan folder induknya, supaya `README.md` langsung tampil di halaman depan.
7. Tulis pesan commit, misalnya `Studi kasus atribusi driver deforestasi`, lalu klik **Commit changes**.

> 💡 File `.gitignore` kadang tersembunyi di Windows/Mac. Kalau tidak ikut ter-upload, tidak masalah, proyek tetap berjalan.

---

## Cara B — Lewat Git (Terminal / Git Bash)

```bash
cd deforestasi-driver-sultra
git init
git add .
git commit -m "Studi kasus atribusi driver deforestasi (Sentinel-1 + Sentinel-2, CNN)"
git branch -M main
git remote add origin https://github.com/keniafeblia/Session-06-Deforestasi.git
git push -u origin main
```

---

