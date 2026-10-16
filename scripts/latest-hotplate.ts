// Quarto pre-render script: copies the newest hotplate/*-hotplate.qmd
// (names sort as YYYY-Qn) to _hotplate-latest.qmd for index.qmd to include.
const names: string[] = [];
for (const entry of Deno.readDirSync("hotplate")) {
  if (entry.isFile && /^\d{4}-Q[1-4]-hotplate\.qmd$/.test(entry.name)) {
    names.push(entry.name);
  }
}
names.sort();
const latest = names.pop();
if (!latest) {
  throw new Error("No hotplate/YYYY-Qn-hotplate.qmd file found");
}
Deno.copyFileSync(`hotplate/${latest}`, "_hotplate-latest.qmd");
console.log(`Latest hotplate: ${latest}`);
