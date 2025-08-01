import fs from 'fs/promises';
import path from 'path';
import { google } from 'googleapis';
import { MongoClient } from 'mongodb';
import dotenv from 'dotenv';
import { OpenAIEmbeddings } from '@langchain/openai';

dotenv.config();

const FOLDER_ID = process.env.GOOGLE_DRIVE_FOLDER_ID;
const TEMP_DIR = './temp';
const embeddings = new OpenAIEmbeddings({
  model: 'text-embedding-3-small'});
const drive = google.drive({
  version: 'v3',
    auth: new google.auth.GoogleAuth({
        keyFile: process.env.GOOGLE_APPLICATION_CREDENTIALS,
        scopes: ['https://www.googleapis.com/auth/drive.readonly'],
    }),
});

async function listFiles() {
    const res = await drive.files.list({
        q: `'${FOLDER_ID}' in parents and trashed = false`,
        fields: 'files(id, name, mimeType, modifiedTime)',
    });
    return res.data.files;
}

async function deleteOldRecords(fileId){
    const client = new MongoClient(process.env.MONGODB_URO);
    await client.connect();
    const db = client.db();
    await db.collection('documents').deleteMany({ file_id: fileId });
    await client.close();
}

async function downloadFile(fileId, filename){
    const destPath = path.join(TEMP_DIR, filename);
    const dest = (await fs.open(destPath, 'w')).createWriteStream();
    const res = await drive.files.get({
        fileId: fileId,
        alt: 'media',
    }, { responseType: 'stream' });
    await new Promise((resolve, reject) => {
        res.data.pipe(dest).on('finish', resolve).on('error', reject);
    });
    return destPath;
}

async function extractTextFromFile(filePath){
    return await fs.readFile(filePath, 'utf-8');
}

function splitTextIntoChunks(text, chunkSize = 1000, overlap = 200) {
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize - overlap) {
        chunks.push(text.slice(i, i + chunkSize));
    }
    return chunks;
}

async function storeChunks(chunks, metadata) {
    const vectors = await embeddings.embedDocuments(chunks);
    const client = new MongoClient(process.env.MONGODB.URI);
    await client.connect();
    const db = client.db();
    const docs = chunks.map((text, i ) => ({
        context: text,
        embeddings: vectors[i],
        ...metadata,
    }));
    await db.collection('documents').insertMany(docs);
    await client.close();
}

async function processFile(file) {
    console.log(`Processing file: ${file.name}`);
    await deleteOldRecords(file.id);
    const filePath = await downloadFile(file.id, file.name);
    const text = await extractTextFromFile(filePath);
    const chunks = splitTextIntoChunks(text);
    await storeChunks(chunks, {
        file_id: file.id,
        file_name: file.name,
        mime_type: file.mime_type,
        modified: file.modified,
    });
    await fs.unlink(filePath);
}

async function main(){
    await fs.mkdir(TEMP_DIR, { recursive: true });
    const files = await listFiles();
    for( const file of files ){
        await processFile(file);
    }
    console.log('Sync Complete');
}

main().catch(console.error);