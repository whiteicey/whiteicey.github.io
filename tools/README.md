# Frontend maintenance tools

## Image pipeline

The production image set lives in `img/optimized/`:

- `hero/` — 1920px WebP, q90, for page/article headers
- `card/` — 800px WebP, q88, for homepage card thumbnails
- `body/` — 2048px WebP, q88, for article body images

Original raster files were backed up outside Git at:

```text
C:\Users\autumn\Desktop\blog\img-originals-backup-2026-09-15\img
```

Keep that backup (or another copy) before running the pruning step.

### Add or change an image

1. Put the original JPG/PNG in `img/`.
2. Reference it normally in front matter (`header-img: img/example.jpg`) or Markdown (`![alt](/img/example.jpg)`).
3. Generate derivatives and rewrite references:

```powershell
npm run images
npm run rewrite-images
npm run audit-images
```

4. Confirm the dry-run report, then remove unreferenced files:

```powershell
python tools/prune_images.py
python tools/prune_images.py --apply
```

## Build and verify

```powershell
npm run build
npm run check
```

`visual_check.py` opens the homepage, a post, About, Tags, and 404 in Chrome. It checks console errors, failed requests, broken images, dark mode, horizontal overflow, and card reveal behavior, and writes screenshots to `tools/visual-check/`.