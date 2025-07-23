import express from 'express';
import bodyParser from 'body-parser';
import { handleRAGChat } from './langchain/agent.js';
import { replyToLine } from './utils/line.js';

const app = express();
app.use(bodyParser.json());

app.post('/webhook/line-bot', async (req, res) => {
  const event = req.body.events?.[0];
  const userId = event?.source?.userId;
  const message = event?.message?.text;
  const replyToken = event?.replyToken;

  const answer = await handleRAGChat({ userId, message });

  await replyToLine({ replyToken, message: answer });

  res.status(200).end();
});

app.listen(3000, () => {
  console.log('LINE webhook listening on port 3000');
});
