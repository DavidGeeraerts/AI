// Quarto pre-render script: points the "Digest (Monthly Release)" link in
// index.qmd at the first entry under "## Digest Catalog" in digest/digest_index.qmd.
const catalogPath = "digest/digest_index.qmd";
const indexPath = "index.qmd";

const catalog = Deno.readTextFileSync(catalogPath).replace(/\r\n/g, "\n");
const section = catalog.split(/^## Digest Catalog\s*$/m)[1];
if (section === undefined) {
  throw new Error(`No "## Digest Catalog" heading found in ${catalogPath}`);
}
const entry = section.split(/^#{1,6}\s/m)[0].match(
  /^\s*[-*+]\s+\[[^\]]*\]\(([^)\s]+?)\.qmd\)/m,
);
if (!entry) {
  throw new Error(`No digest link found under "## Digest Catalog" in ${catalogPath}`);
}
const href = `${entry[1]}.html`;

const original = Deno.readTextFileSync(indexPath);
const linkPattern = /\[Read the latest Digest\]\([^)]*\)/;
if (!linkPattern.test(original)) {
  throw new Error(`No "Read the latest Digest" link found in ${indexPath}`);
}
const updated = original.replace(
  linkPattern,
  () => `[Read the latest Digest](${href})`,
);

// Skip identical writes so preview file watchers don't see a change.
if (updated !== original) {
  Deno.writeTextFileSync(indexPath, updated);
}
console.log(`Latest digest: ${href}`);
