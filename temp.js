import fs from "fs";

console.log("Reading oauth-client.json...");
const data = fs.readFileSync("oauth-client.json");
console.log("File content:", data.toString());
