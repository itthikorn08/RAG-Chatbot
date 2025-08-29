import fs from "fs";
import path from "path";
import dotenv from "dotenv";
import { google } from "googleapis";
import cron from "node-cron";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import mammoth from "mammoth";
import XLSX from "xlsx";

// --- เปลี่ยนแปลงตรงนี้ ---
import { QdrantClient } from "@qdrant/js-client-rest";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { CharacterTextSplitter } from "@langchain/textsplitters";

dotenv.config();

const COLLECTION_NAME = "gemini_documents";

// --- เปลี่ยนการเชื่อมต่อจาก MongoDB เป็น Qdrant ---
const qdrantClient = new QdrantClient({
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
    checkCompatibility: false,
});

// --- Embedding Model ---
const embeddings = new GoogleGenerativeAIEmbeddings({
    model: "gemini-embedding-001",
    apiKey: process.env.GOOGLE_API_KEY,
});

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

/**
 * โหลดไฟล์จาก Google Drive และส่ง buffer กลับมา
 * ถ้าเป็น Google Docs/Sheets จะ export เป็น format ที่เหมาะสม
 */
async function downloadFileBuffer(fileId, mimeType) {
    console.time("downloadFileBuffer");
    let exportMimeType = null;

    if (mimeType === "application/vnd.google-apps.document") {
        exportMimeType = "text/plain"; // Google Docs → Plain Text
    } else if (mimeType === "application/vnd.google-apps.spreadsheet") {
        exportMimeType =
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"; // Google Sheets → XLSX
    }

    let buffer;
    if (exportMimeType) {
        const res = await drive.files.export(
            { fileId, mimeType: exportMimeType },
            { responseType: "arraybuffer" }
        );
        buffer = Buffer.from(res.data);
    } else {
        const res = await drive.files.get(
            { fileId, alt: "media" },
            { responseType: "arraybuffer" }
        );
        buffer = Buffer.from(res.data);
    }
    console.timeEnd("downloadFileBuffer");
    return buffer;
}

/**
 * แปลง buffer + mimeType เป็นข้อความ
 */
async function bufferToText(buffer, mimeType) {
    console.time("bufferToText");
    // Google Docs (plain text)
    if (mimeType === "application/vnd.google-apps.document") {
        console.timeEnd("bufferToText");
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
        console.timeEnd("bufferToText");
        return text;
    }

    // PDF → ใช้ LangChain PDFLoader
    if (mimeType === "application/pdf") {
        const tempPath = path.join(process.cwd(), `temp_${Date.now()}.pdf`);
        fs.writeFileSync(tempPath, buffer);

        const loader = new PDFLoader(tempPath);
        const docs = await loader.load();

        fs.unlinkSync(tempPath);
        console.timeEnd("bufferToText");
        return docs.map(doc => doc.pageContent).join("\n");
    }

    // Word
    if (
        mimeType ===
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
        mimeType === "application/msword"
    ) {
        const result = await mammoth.extractRawText({ buffer });
        console.timeEnd("bufferToText");
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
        console.timeEnd("bufferToText");
        return text;
    }

    // Text file
    if (mimeType.startsWith("text/")) {
        console.timeEnd("bufferToText");
        return buffer.toString("utf8");
    }

    console.warn(`⚠️ Unsupported file type: ${mimeType}`);
    console.timeEnd("bufferToText");
    return "";
}

// --- Process a file ---
async function processFile(fileId, fileName, mimeType) {
    console.time(`processFile: ${fileName}`);
    console.log(`📄 Processing: ${fileName} (${mimeType})`);

    // --- เปลี่ยนการลบข้อมูลเก่าจาก MongoDB เป็น Qdrant ---
    console.time("deleteOldData");
    try {
        await qdrantClient.delete(COLLECTION_NAME, {
            filter: {
                must: [
                    {
                        key: "file_id",
                        match: {
                            value: fileId,
                        },
                    },
                ],
            },
        });
    } catch (e) {
        console.warn(`⚠️ Could not delete old points for file ID: ${fileId}. It might not exist yet.`);
    }
    console.timeEnd("deleteOldData");

    // โหลดไฟล์
    const buffer = await downloadFileBuffer(fileId, mimeType);
    const text = await bufferToText(buffer, mimeType);
    if (!text.trim()) {
        console.warn(`⚠️ No text extracted from ${fileName}`);
        console.timeEnd(`processFile: ${fileName}`);
        return;
    }

    // Split
    console.time("textSplitter");
    const splitter = new CharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 300,
    });
    const docs = await splitter.createDocuments([text]);
    console.timeEnd("textSplitter");

    // Vector Store
    console.time("addDocumentsToVectorStore");
    // --- เปลี่ยนการสร้าง Vector Store จาก MongoDB เป็น Qdrant ---
    const vectorStore = new QdrantVectorStore(embeddings, {
        url: process.env.QDRANT_URL,
        apiKey: process.env.QDRANT_API_KEY,
        collectionName: COLLECTION_NAME,
    });

    await vectorStore.addDocuments(
        docs.map((d) => ({
            pageContent: d.pageContent,
            metadata: { file_id: fileId, file_name: fileName, mimeType },
        }))
    );
    console.timeEnd("addDocumentsToVectorStore");

    console.log(`✅ Done: ${fileName}`);
    console.timeEnd(`processFile: ${fileName}`);
}

// --- Check folder for updates ---
let processedFiles = new Map(); // fileId -> modifiedTime

async function checkDriveFolder() {
    console.time("checkDriveFolder");
    const res = await drive.files.list({
        q: `'${process.env.GOOGLE_DRIVE_FOLDER_ID}' in parents and trashed=false`,
        fields: "files(id, name, mimeType, modifiedTime)",
    });

    const filesToProcess = [];
    for (const file of res.data.files) {
        const lastModified = new Date(file.modifiedTime).getTime();
        if (
            !processedFiles.has(file.id) ||
            processedFiles.get(file.id) < lastModified
        ) {
            filesToProcess.push(file);
        }
    }

    for (const file of filesToProcess) {
        await processFile(file.id, file.name, file.mimeType);
        processedFiles.set(file.id, new Date(file.modifiedTime).getTime());
    }

    console.timeEnd("checkDriveFolder");
}

// --- Run every minute ---
cron.schedule("* * * * *", async () => {
    console.log("🔍 Checking Google Drive...");
    await checkDriveFolder();
});

console.log("🚀 Drive watcher (OAuth2) started...");