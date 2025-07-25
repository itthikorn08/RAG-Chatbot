// memory.js
import { MongoClient } from 'mongodb';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import * as dotenv from 'dotenv';
import dayjs from 'dayjs';
dotenv.config();

let mongoClient;
let _cachedDb; // Cache for the database instance

// กำหนดจำนวนข้อความสูงสุดที่ "เก็บในฐานข้อมูล"
const MAX_MESSAGES_IN_DB = 20;

export class MongoChatMessageHistoryManual {
  constructor({ collection, sessionId }) {
    this.collection = collection;
    this.sessionId = sessionId;
  }

  /**
   * Retrieves messages for the current session from MongoDB.
   * NOTE: This method will retrieve ALL messages up to MAX_MESSAGES_IN_DB.
   * The actual trimming for LLM context will happen in handleRAGChat.
   * @returns {Promise<Array<AIMessage|HumanMessage>>} An array of LangChain message objects.
   */
  async getMessages() {
    try {
      const doc = await this.collection.findOne({ sessionId: this.sessionId });
      const history = doc?.history ?? [];
      // console.log(`[Memory - ${this.sessionId}] getMessages: Retrieved ${history.length} messages from DB.`);

      return history.map(msg => {
        if (msg.role === 'human') return new HumanMessage(msg.content);
        if (msg.role === 'ai') return new AIMessage(msg.content);
        throw new Error(`Unknown message role found in history: ${msg.role}`);
      });
    } catch (error) {
      console.error(`[Memory - ${this.sessionId}] Error getting messages:`, error);
      return []; 
    }
  }

  async addUserMessage(message) {
    await this._appendMessage({ role: 'human', content: message });
  }

  async addAIMessage(message) {
    await this._appendMessage({ role: 'ai', content: message });
  }

  async addMessages(messages) {
    // console.log(`[Memory - ${this.sessionId}] addMessages: Processing ${messages.length} messages from LangChain.`);
    for (const msg of messages) {
      if (msg._getType() === 'human' || msg._getType() === 'ai') {
        await this._appendMessage({ role: msg._getType(), content: msg.content });
      } else {
        console.warn(`[Memory - ${this.sessionId}] Skipping unknown message type in addMessages: ${msg._getType()}`);
      }
    }
  }

  async clear() {
    try {
      await this.collection.updateOne(
        { sessionId: this.sessionId },
        { $set: { history: [] } }
      );
      console.log(`[Memory - ${this.sessionId}] clear: Chat history cleared.`);
    } catch (error) {
      console.error(`[Memory - ${this.sessionId}] Error clearing history:`, error);
    }
  }

  async _appendMessage(msg) {
    msg.timestamp = dayjs().toISOString();
    const currentTimeStamp = dayjs().toDate();

    try {
      await this.collection.updateOne(
        { sessionId: this.sessionId },
        {
          $push: { history: msg },
          $set: { last_activity_timestamp: currentTimeStamp }
        },
        { upsert: true }
      );

      
      const doc = await this.collection.findOne(
        { sessionId: this.sessionId },
        { projection: { history: { $slice: -(MAX_MESSAGES_IN_DB + 1) } } }
      );
      const history = doc?.history ?? [];

      if (history.length > MAX_MESSAGES_IN_DB) {
        const trimmed = history.slice(-MAX_MESSAGES_IN_DB); 
        await this.collection.updateOne(
          { sessionId: this.sessionId },
          { $set: { history: trimmed } }
        );
        // console.log(`[Memory - ${this.sessionId}] _appendMessage: History trimmed to ${trimmed.length} messages in DB.`);
      }
    } catch (error) {
      console.error(`[Memory - ${this.sessionId}] Error appending/trimming message:`, error);
    }
  }
}

/**
 * Connects to MongoDB and returns the database instance.
 * Ensures a single connection and reuses it across calls.
 * @returns {Promise<Db>} The MongoDB database instance.
 * @throws {Error} If MONGODB_URI or MONGODB_NAME environment variables are missing, or if connection fails.
 */
export async function connectToMongoDB() {
  if (_cachedDb) {
    return _cachedDb;
  }

  const uri = process.env.MONGODB_URI;
  const dbName = process.env.MONGODB_NAME;

  if (!uri || !dbName) {
    
    throw new Error('Missing MONGODB_URI or MONGODB_NAME environment variables. Please check your .env file.');
  }

  mongoClient = new MongoClient(uri);

  try {
    console.log('Connecting to MongoDB...');
    await mongoClient.connect();
    _cachedDb = mongoClient.db(dbName);
    console.log('MongoDB connected successfully!');
    return _cachedDb;
  } catch (error) {
    console.error('Failed to connect to MongoDB:', error);
    
    throw error; 
  }
}

/**
 * Gets the chat memory instance for a specific user.
 * This function will ensure MongoDB connection and return the memory object.
 * @param {string} userId - The ID of the user.
 * @returns {Promise<{chatHistory: MongoChatMessageHistoryManual}>} An object containing the chatHistory instance.
 * @throws {Error} If there's an issue connecting to MongoDB or initializing memory.
 */
export async function getMemoryForUser(userId) {
  try {
    const db = await connectToMongoDB(); // This call can now throw an error
    const collection = db.collection('chat_histories');
    return {
      chatHistory: new MongoChatMessageHistoryManual({ collection, sessionId: userId }),
    };
  } catch (error) {
    
    console.error(`[getMemoryForUser] Error initializing memory for user ${userId}:`, error);
    throw error; 
  }
}

export async function createTtlIndex() {
  const db = await connectToMongoDB();
  const collection = db.collection('chat_histories');
  const TTL_SECONDS = 5 * 60; // 5 นาที = 300 วินาที

  try {
    // ตรวจสอบว่ามี Index ชื่อ 'last_activity_timestamp_ttl' อยู่แล้วหรือไม่
    const indexes = await collection.indexes();
    const ttlIndexExists = indexes.some(index => index.name === 'last_activity_timestamp_ttl');

    if (!ttlIndexExists) {
      // สร้าง TTL Index บนฟิลด์ 'last_activity_timestamp'
      // expireAfterSeconds คือระยะเวลาที่เอกสารจะถูกลบหลังจาก timestamp ในฟิลด์นั้น
      await collection.createIndex(
        { "last_activity_timestamp": 1 },
        { expireAfterSeconds: TTL_SECONDS, name: 'last_activity_timestamp_ttl' }
      );
      console.log(`TTL index 'last_activity_timestamp_ttl' created on 'chat_histories' collection, expiring after ${TTL_SECONDS} seconds.`);
    } else {
      console.log(`TTL index 'last_activity_timestamp_ttl' already exists.`);
    }
  } catch (error) {
    console.error("Error creating TTL index:", error);
    // หาก Index มีอยู่แล้วแต่มี expireAfterSeconds ที่ต่างกัน MongoDB จะโยน Error
    // ซึ่งควรถูกจัดการหรือตรวจสอบล่วงหน้าใน Production
  }
}