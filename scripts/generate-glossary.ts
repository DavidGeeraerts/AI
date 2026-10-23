const sourcePath = "appendices/Glossary.qmd";
const outputPath = "appendices/glossary.yml";
const source = Deno.readTextFileSync(sourcePath);
const lines = source.replace(/\r\n/g, "\n").split("\n");
const glossary = new Map<string, string[]>();
let currentTerm: string | undefined;
let definitionLines: string[] = [];
let inFrontMatter = lines[0]?.trim() === "---";

function cleanInlineMarkdown(text: string): string {
  return text
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/(`+)(.*?)\1/g, "$2")
    .replace(/(\*\*|__)(.*?)\1/g, "$2")
    .replace(/(\*|_)(.*?)\1/g, "$2")
    .replace(/<\/?[^>]+>/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function saveDefinition(): void {
  if (currentTerm === undefined) {
    return;
  }

  const definition = cleanInlineMarkdown(
    definitionLines
      .map((line) => line
        .replace(/^\s*>+\s?/, "")
        .replace(/^\s*#{1,6}\s+/, "")
        .replace(/^\s*(?:[-*+]|\d+[.)])\s+/, "")
        .trim())
      .filter((line) => line.length > 0 && !/^---+$/.test(line))
      .join(" "),
  );

  if (!definition) {
    throw new Error(`Glossary term has no definition: ${currentTerm}`);
  }

  const definitions = glossary.get(currentTerm) ?? [];
  if (!definitions.includes(definition)) {
    definitions.push(definition);
  }
  glossary.set(currentTerm, definitions);
}

for (const line of lines) {
  if (inFrontMatter) {
    if (line.trim() === "---") {
      inFrontMatter = false;
    }
    continue;
  }

  const heading = line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/);
  if (heading && heading[1].length <= 3) {
    saveDefinition();
    currentTerm = heading[1].length === 3
      ? cleanInlineMarkdown(heading[2])
      : undefined;
    definitionLines = [];
    continue;
  }

  if (currentTerm !== undefined) {
    definitionLines.push(line);
  }
}
saveDefinition();

if (glossary.size === 0) {
  throw new Error(`No glossary entries found in ${sourcePath}`);
}

const yaml = [...glossary.entries()]
  .map(([term, definitions]) => {
    const definition = definitions.join(" ");
    return `${JSON.stringify(term)}: ${JSON.stringify(definition)}`;
  })
  .join("\n") + "\n";

Deno.writeTextFileSync(outputPath, yaml);
console.log(`Generated ${glossary.size} glossary entries in ${outputPath}`);
