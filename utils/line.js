import axios from 'axios';
import * as dotenv from 'dotenv';
dotenv.config();

export async function replyToLine({ replyToken, message }) {
  await axios.post(
    'https://api.line.me/v2/bot/message/reply',
    {
      replyToken,
      messages: [
        {
          type: 'text',
          text: message,
        },
      ],
    },
    {
      headers: {
        Authorization: `Bearer ${process.env.LINE_CHANNEL_ACCESS_TOKEN}`,
        'Content-Type': 'application/json',
      },
    }
  );
}
