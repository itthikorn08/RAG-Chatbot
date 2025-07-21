import express from 'express';
import { middleware, Client } from '@line/bot-sdk';
import dotenv from 'dotenv';
import { ChatOpenAI } from '@langchain/openai';
import { getCollection } from './Database/mongo.js';


dotenv.config();


const config = {
    channelAccessToken: process.env.LINE_CHANNEL_ACCESS_TOKEN,
    channelSecret: process.env.LINE_CHANNEL_SECRET
};


const lineClient = new Client(config);
const app = express();


const model = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0.7,
});


app.use(express.json({
    verify: (req, res, buf) => {
        req.rawBody = buf.toString();
    }
}));


app.use(middleware(config));

/**
 * @param {string} text - The user's message.
 * @returns {Promise<string>} The AI's response.
 */

const handleUserMessage = async (text) => {
    try {
        const response = await model.invoke(text);

        return response?.content || response?.text || "Sorry, I didn't understand that. Please try rephrasing.";
    } catch (error) {
        console.error("Error invoking OpenAI model:", error);
        return "Oops! There was an error processing your request. Please try again later.";
    }
};


app.post('/webhook', async (req, res) => {

    const chatCollection = await getCollection('chat_history');
    console.log('Chat collection initialized');
    const events = req.body.events;


    for (const event of events) {

        if (event.type === 'message' && event.message.type === 'text') {
            const userMessage = event.message.text;
            console.log(`Received message from user: ${userMessage}`);


            const reply = await handleUserMessage(userMessage);


            try {
                await lineClient.replyMessage(event.replyToken, {
                    type: 'text',
                    text: reply,
                });
                console.log(`Replied to user: ${reply}`);

                await chatCollection.updateOne(
                    { userId: event.source.userId },
                    {
                        $push: {
                            history: {
                                $each: [{
                                    timestamp: new Date(),
                                    userMessage: userMessage,
                                    replyMessage: reply,
                                }],
                                $slice: -20
                            }
                        }
                    },
                    { upsert: true } // ถ้ายังไม่มี userId นี้ ให้สร้าง document ใหม่
                );

                console.log('Chat history saved to MongoDB');
            } catch (replyError) {
                console.error("Error replying to LINE message:", replyError);
            }
        }
    }

    res.status(200).send('OK');
});


const PORT = process.env.PORT || 3000;


app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
    console.log('Ensure your LINE Channel Access Token and Channel Secret are set in .env');
});
