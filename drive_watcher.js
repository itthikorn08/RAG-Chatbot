import fs from "fs";
import path from "path";
import dotenv from "dotenv";
import { google } from "googleapis";
import cron from "node-cron";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import mammoth from "mammoth";
import XLSX from "xlsx";
import { MongoClient } from "mongodb";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MongoDBAtlasVectorSearch } from "@langchain/community/vectorstores/mongodb_atlas";
import { CharacterTextSplitter } from "@langchain/textsplitters";

dotenv.config();

const uri = process.env.MONGODB_URI;
const dbName = process.env.MONGODB_NAME;

// --- Auth with OAuth2 ---
const credentials = JSON.parse(fs.readFileSync("oauth-client.json"));
const token = JSON.parse(fs.readFileSync("token.json"));
const { client_secret, client_id, redirect_uris } = credentials.installed;
const oAuth2Client = new google.auth.OAuth2(
    client_id,
    client_secret,
    redirect_uris[0]
);
oAuth2Client.setCredentials(token);
const drive = google.drive({ version: "v3", auth: oAuth2Client });

// --- MongoDB ---
const mongoClient = new MongoClient(uri);
await mongoClient.connect();
const collection = mongoClient.db(dbName).collection("gemini_documents");

// --- Embedding Model ---
const embeddings = new GoogleGenerativeAIEmbeddings({
    model: "gemini-embedding-001",
    apiKey: process.env.GOOGLE_API_KEY,
});

/**
 * โหลดไฟล์จาก Google Drive และส่ง buffer กลับมา
 * ถ้าเป็น Google Docs/Sheets จะ export เป็น format ที่เหมาะสม
 */
async function downloadFileBuffer(fileId, mimeType) {
    let exportMimeType = null;

    if (mimeType === "application/vnd.google-apps.document") {
        exportMimeType = "text/plain"; // Google Docs → Plain Text
    } else if (mimeType === "application/vnd.google-apps.spreadsheet") {
        exportMimeType =
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"; // Google Sheets → XLSX
    }

    if (exportMimeType) {
        const res = await drive.files.export(
            { fileId, mimeType: exportMimeType },
            { responseType: "arraybuffer" }
        );
        return Buffer.from(res.data);
    } else {
        const res = await drive.files.get(
            { fileId, alt: "media" },
            { responseType: "arraybuffer" }
        );
        return Buffer.from(res.data);
    }
}

/**
 * แปลง buffer + mimeType เป็นข้อความ
 */
async function bufferToText(buffer, mimeType) {
    // Google Docs (plain text)
    if (mimeType === "application/vnd.google-apps.document") {
        return buffer.toString("utf8");
    }

    // Google Sheets (XLSX)
    if (mimeType === "application/vnd.google-apps.spreadsheet") {
        const workbook = XLSX.read(buffer, { type: "buffer" });
        let text = "";
        workbook.SheetNames.forEach((sheetName) => {
            const sheet = XLSX.utils.sheet_to_csv(workbook.Sheets[sheetName]);
            text += sheet + "\n";
        });
        return text;
    }

    // PDF → ใช้ LangChain PDFLoader
    if (mimeType === "application/pdf") {
        const tempPath = path.join(process.cwd(), `temp_${Date.now()}.pdf`);
        fs.writeFileSync(tempPath, buffer);

        const loader = new PDFLoader(tempPath);
        const docs = await loader.load();

        fs.unlinkSync(tempPath);

        return docs.map(doc => doc.pageContent).join("\n");
    }

    // Word
    if (
        mimeType ===
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
        mimeType === "application/msword"
    ) {
        const result = await mammoth.extractRawText({ buffer });
        return result.value;
    }

    // Excel
    if (
        mimeType ===
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" ||
        mimeType === "application/vnd.ms-excel"
    ) {
        const workbook = XLSX.read(buffer, { type: "buffer" });
        let text = "";
        workbook.SheetNames.forEach((sheetName) => {
            const sheet = XLSX.utils.sheet_to_csv(workbook.Sheets[sheetName]);
            text += sheet + "\n";
        });
        return text;
    }

    // Text file
    if (mimeType.startsWith("text/")) {
        return buffer.toString("utf8");
    }

    console.warn(`⚠️ Unsupported file type: ${mimeType}`);
    return "";
}

// --- Process a file ---
async function processFile(fileId, fileName, mimeType) {
    console.log(`📄 Processing: ${fileName} (${mimeType})`);

    // ลบข้อมูลเก่า
    await collection.deleteMany({ file_id: fileId });

    // โหลดไฟล์
    const buffer = await downloadFileBuffer(fileId, mimeType);
    const text = await bufferToText(buffer, mimeType);
    if (!text.trim()) {
        console.warn(`⚠️ No text extracted from ${fileName}`);
        return;
    }

    // Split
    const splitter = new CharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 300,
    });
    const docs = await splitter.createDocuments([text]);

    // Vector Store
    const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
        collection,
        indexName: "vector_index_gemini",
        textKey: "text",
        embeddingKey: "embedding",
    });

    await vectorStore.addDocuments(
        docs.map((d) => ({
            pageContent: d.pageContent,
            metadata: { file_id: fileId, file_name: fileName, mimeType },
        }))
    );

    console.log(`✅ Done: ${fileName}`);
}

// --- Check folder for updates ---
let processedFiles = new Map(); // fileId -> modifiedTime

async function checkDriveFolder() {
    const res = await drive.files.list({
        q: `'${process.env.GOOGLE_DRIVE_FOLDER_ID}' in parents and trashed=false`,
        fields: "files(id, name, mimeType, modifiedTime)",
    });

    for (const file of res.data.files) {
        const lastModified = new Date(file.modifiedTime).getTime();
        if (
            !processedFiles.has(file.id) ||
            processedFiles.get(file.id) < lastModified
        ) {
            await processFile(file.id, file.name, file.mimeType);
            processedFiles.set(file.id, lastModified);
        }
    }
}

// --- Run every minute ---
cron.schedule("* * * * *", async () => {
    console.log("🔍 Checking Google Drive...");
    await checkDriveFolder();
});

console.log("🚀 Drive watcher (OAuth2) started...");
