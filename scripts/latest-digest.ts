// Quarto pre-render script: copies the newest digest/*-digest.qmd
// (names sort as YYYY-Qn) to _digest-latest.qmd for index.qmd to include.
const names: string[] = [];
for (const entry of Deno.readDirSync("digest")) {
  if (entry.isFile && /^\d{4}-Q[1-4]-digest\.qmd$/.test(entry.name)) {
    names.push(entry.name);
  }
}
names.sort();
const latest = names.pop();
if (!latest) {
  throw new Error("No digest/YYYY-Qn-digest.qmd file found");
}
Deno.copyFileSync(`digest/${latest}`, "_digest-latest.qmd");
console.log(`Latest digest: ${latest}`);
