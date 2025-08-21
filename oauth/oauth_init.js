import fs from "fs";
import readline from "readline";
import { google } from "googleapis";

console.log("🚀 Starting OAuth2 init...");

const SCOPES = ["https://www.googleapis.com/auth/drive.readonly"];
const CREDENTIALS_PATH = "oauth-client.json";
const TOKEN_PATH = "token.json";

function authorize() {
  try {
    if (!fs.existsSync(CREDENTIALS_PATH)) {
      console.error(`❌ File ${CREDENTIALS_PATH} not found`);
      process.exit(1);
    }
    console.log(`📄 Reading ${CREDENTIALS_PATH}...`);
    const credentials = JSON.parse(fs.readFileSync(CREDENTIALS_PATH));
    const { client_secret, client_id, redirect_uris } = credentials.installed;
    const oAuth2Client = new google.auth.OAuth2(
      client_id,
      client_secret,
      redirect_uris[0]
    );

    const authUrl = oAuth2Client.generateAuthUrl({
      access_type: "offline",
      scope: SCOPES,
    });
    console.log("🌐 Authorize this app by visiting this URL:\n", authUrl);

    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    rl.question("Enter the code from that page: ", (code) => {
      rl.close();
      oAuth2Client.getToken(code, (err, token) => {
        if (err) return console.error("Error retrieving token", err);
        oAuth2Client.setCredentials(token);
        fs.writeFileSync(TOKEN_PATH, JSON.stringify(token));
        console.log(`✅ Token stored to ${TOKEN_PATH}`);
      });
    });
  } catch (err) {
    console.error("❌ Error:", err);
  }
}

authorize();
