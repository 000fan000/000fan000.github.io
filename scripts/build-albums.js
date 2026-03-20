const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const jpgsDir = path.join(root, 'photography', 'jpgs');
const outFile = path.join(root, 'photography', 'albums.json');

function isImage(name) {
  const lower = name.toLowerCase();
  return lower.endsWith('.jpg') || lower.endsWith('.jpeg');
}

function buildAlbums() {
  if (!fs.existsSync(jpgsDir)) {
    console.error(`Missing directory: ${jpgsDir}`);
    process.exit(1);
  }

  const folders = fs.readdirSync(jpgsDir, { withFileTypes: true })
    .filter((dirent) => dirent.isDirectory())
    .map((dirent) => dirent.name)
    .sort((a, b) => a.localeCompare(b, 'zh-Hans-CN'));

  const albums = {};

  folders.forEach((folder) => {
    const folderPath = path.join(jpgsDir, folder);
    const files = fs.readdirSync(folderPath)
      .filter(isImage)
      .sort((a, b) => a.localeCompare(b, 'en'));
    if (files.length > 0) {
      albums[folder] = files;
    }
  });

  fs.writeFileSync(outFile, JSON.stringify(albums, null, 2));
  console.log(`Wrote ${Object.keys(albums).length} albums to ${outFile}`);
}

buildAlbums();
