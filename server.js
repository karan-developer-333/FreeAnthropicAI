import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { BypassedClaudeClient } from './bypassed_anthropic.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3006;

// 1. Enable Full CORS Support for cross-origin applications and APIs
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(204);
  }
  next();
});

// 2. Parse JSON payloads and serve the stunning static UI assets at root
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// 3. Expose high-speed chat completion bypass endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const {
      thinking = false,
      streaming = false,
      model,
      freemodelapi,
      messages = [],
      system,
      max_tokens,
      temperature
    } = req.body;

    console.log(`[Merged Server] POST /api/chat -> Model: ${model}, Streaming: ${streaming}, Thinking: ${thinking}`);

    // Initialize our bypassed client using the custom API key passed in body if present
    const client = new BypassedClaudeClient({
      apiKey: freemodelapi
    });

    if (streaming) {
      // SSE Streaming Response - pipe decompressed upstream chunks instantly with zero lag
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no'
      });

      try {
        const stream = await client.create({
          model,
          messages,
          system,
          max_tokens,
          temperature,
          stream: true,
          thinking
        });

        stream.pipe(res);
      } catch (streamErr) {
        console.error('[Merged Server] Stream pipeline error:', streamErr);
        res.write(`data: ${JSON.stringify({ error: streamErr.message })}\n\n`);
        res.end();
      }
    } else {
      // Standard high-speed JSON Response
      try {
        const result = await client.create({
          model,
          messages,
          system,
          max_tokens,
          temperature,
          stream: false,
          thinking
        });

        res.status(200).json(result);
      } catch (jsonErr) {
        console.error('[Merged Server] JSON request creation error:', jsonErr);
        res.status(500).json({ error: jsonErr.message });
      }
    }
  } catch (err) {
    console.error('[Merged Server] Unexpected request processor error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Start unified server
app.listen(PORT, () => {
  console.log(`\n=============================================================`);
  console.log(`🤖 Unified Claude CLI Bypass Server running at:`);
  console.log(`👉 http://localhost:${PORT}`);
  console.log(`👉 UI Playground: http://localhost:${PORT}`);
  console.log(`👉 API Endpoint:  http://localhost:${PORT}/api/chat`);
  console.log(`=============================================================\n`);
});
