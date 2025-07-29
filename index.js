import express from 'express';
import bodyParser from 'body-parser';
import { handleRAGChat } from './langchain/agent.js';
import { replyToLine } from './utils/line.js';
import { createTtlIndex } from './langchain/memory.js';

const app = express();
app.use(bodyParser.json());

const PORT = process.env.PORT || 3000;

async function startServer() {
  try {
    await createTtlIndex();
    console.log('MongoDB TTL index created successfully!');

    app.listen(PORT, () => {
      console.log(`LINE webhook listening on port ${PORT}`);
    });
  } catch (error) {
    console.error('Failed to create TTL index:', error);
    process.exit(1);
  }
}

startServer();

app.post('/webhook/line-bot', async (req, res) => {
  const event = req.body.events?.[0];
  const userId = event?.source?.userId;
  const message = event?.message?.text;
  const replyToken = event?.replyToken;

  const answer = await handleRAGChat({ userId, message });

  await replyToLine({ replyToken, message: answer });

  res.status(200).end();
});


