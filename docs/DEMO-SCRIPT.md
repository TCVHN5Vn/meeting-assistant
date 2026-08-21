# Demo script

A four-minute walkthrough. Everything in **bold quotes** is said out loud;
everything in `code` is typed or clicked.

Every question here has been checked against the current index, so none of
them will come up empty on camera.

---

## Before you hit record

```bash
brew services start ollama                 # if it is not already running
cd ~/meeting-assistant-folder
source .venv/bin/activate
uvicorn app.server:app --reload            # wait for "Server ready."
```

- Open **http://localhost:8000**, hard-reload with **Cmd+Shift+R**, sign in.
- Make the window **wide** — under 900px the two panes stack.
- Ask one throwaway question first so the embedding model is warm. The first
  question of the session is slow; the rest are not.
- Test your microphone once. If the transcript garbles your voice, move
  closer and slow down — the whole demo rests on it.

**Two things about pacing.** Nothing is transcribed until you **pause** — so
finish each sentence and stop for a beat. And answers take 10–20 seconds to
complete on a local model, so the script gives you something to say while
each one writes. Cut those pauses in the edit if you want it tighter.

---

## 1 · Open — what it is  *(~25s, not recording)*

> **"This is a meeting assistant. It listens to a meeting, writes down what
> is said, answers questions while the meeting is still going, and pulls out
> the action items people committed to."**
>
> **"The part I care about is that all of it runs on this laptop. The
> speech recognition, the search, and the language model are all local.
> There is no API key in this project, and no audio or document ever leaves
> the machine — which for a tool that records meetings is a compliance
> argument, not just a cost saving."**

---

## 2 · Live transcription  *(~40s)*

`Click` **Start recording** — allow the microphone if asked.

Wait for the red dot, then role-play a short meeting. **Pause after each
sentence.**

> **"So the next item is the notice period for general meetings."**
>
> *(pause)*
>
> **"I think we need to check what the policy actually says before we send
> anything out."**

`Point at` the transcript as it appears.

> **"That is appearing as I speak. And notice it arrives in bursts rather
> than continuously — it waits until I stop talking before transcribing.
> That is deliberate. Cutting audio on a stopwatch slices words in half;
> cutting at natural pauses gives you clean sentences. It measurably
> improved transcription quality when I changed it."**

---

## 3 · Asking out loud — the main moment  *(~60s)*

Still recording. Say the wake phrase **clearly and a little slowly** —
it is the shortest part of the sentence and the easiest to mishear.

> **"Hey assistant — what is the notice period for a general meeting?"**
>
> *(pause, and stop talking)*

While the sources appear and the answer starts writing:

> **"I did not touch the keyboard. It heard its name, worked out that the
> rest of the sentence was the question, searched the company documents, and
> is answering now — and transcription has not stopped while it does it. The
> meeting does not pause for the assistant."**

When the answer finishes, `point at` the sources.

> **"And it cites what it used. That teal chip is a written document, the
> amber ones are previous meetings, and the number is how close the match
> was. It is answering from our governance policy — fourteen to sixty days."**

---

## 4 · One question, two kinds of source  *(~50s)*

`Type` in the Ask box:

```
what does the policy say about volunteers?
```

> **"This one is more interesting, because the answer is in two places. The
> written policy says what we are supposed to do about volunteers, and a
> previous meeting has people actually discussing it."**

When the sources appear:

> **"Both come back in one ranked list — the document and the meeting,
> interleaved. That only works because they live in the same search index.
> If I had kept them in two separate indexes, the scores would not be
> comparable and I would have had to invent some rule for merging them."**
>
> **"And look at the meeting citation — it is a timestamp. I can open the
> recording and go listen to that exact moment. A citation you cannot check
> is decoration."**

---

## 5 · What it does when it does not know  *(~35s)*

> **"The thing I actually care most about is what happens when the answer
> is not there."**

`Type`:

```
what is our stock price?
```

> **"Nothing. No sources, and it says so."**
>
> **"That is not the model being modest. Nothing scored above the relevance
> threshold, so the language model was never called at all. It cannot invent
> an answer to a question it was never asked to write. That is a rule in
> code, not an instruction in a prompt — and a prompt is a request, not a
> guarantee."**

---

## 6 · Stopping, and what changes  *(~40s)*

Before stopping, say one clearly quotable commitment. **Pause afterwards.**

> **"Right — we agreed that Karen will circulate the revised budget to the
> members before Friday."**
>
> *(pause)*

`Click` **Stop recording**. `Point at` the status line.

> **"Watch the status. The transcript was saved the whole time, but saved
> and searchable are different things. It is grouping what I just said into
> passages, embedding them, and adding them to the index. Now it says
> searchable — and only now can I ask about it."**

`Type`:

```
what did we agree about the budget?
```

> **"And there it is, cited back to my own meeting, seconds after I said it."**

---

## 7 · Action items  *(~50s)*

`Click` the **ACTION ITEMS** tab, then **Extract action items**.

> **"This is the part I had to be most careful about. Asking a model to
> find action items works fine — the problem is it works just as well when
> there are none. Meetings are full of things that sound like commitments
> and are not."**

While it runs:

> **"So every task has to come with the sentence it came from, copied
> exactly, and the code checks that sentence really is in the transcript
> before it stores anything. A model that invents a task has to invent the
> quote too, and an invented quote is not there to find."**

When the task appears, `point at` the italic quote.

> **"Karen, circulate the budget, before Friday — and underneath, the exact
> words I said, with the timestamp. I can check every line of this against
> the recording."**
>
> **"For what it's worth, I measured this. On a meeting that contained no
> action items at all, a naive prompt returned thirty-three of them. This
> returned zero."**

---

## 8 · Close  *(~25s)*

> **"So: speech in, transcript stored, questions answered out loud from both
> documents and past meetings with citations you can follow, and action items
> that are checked rather than trusted."**
>
> **"All of it on one laptop, with no API key and nothing leaving the
> machine. Around six thousand lines of Python, a hundred and twenty tests,
> and the interesting parts were not the AI calls — they were the boundaries
> around them. Deciding where to cut the audio, when to refuse to answer, and
> how to verify what the model gives you back."**

---

## If something goes wrong on camera

| Problem | What to do |
|---|---|
| Wake phrase ignored | Look at the transcript. If the word after "hey" is mangled, say it again more separated: *"Hey… assistant… what is…"* |
| Nothing transcribes | You have not paused. Stop talking for a full second. |
| Answer is slow | Expected — 10–20s locally. Keep narrating. |
| Panes stacked vertically | Window is under 900px. Widen it. |
| "could not find" about your own meeting | Status has not reached **searchable** yet. Wait. |

**Record each section separately if it's easier.** Every one stands alone,
and cutting the model's thinking time out makes the final video much tighter
than doing it in one take.

---

## Shorter version (90 seconds)

If you need a cut-down demo, keep only these:

1. Start recording, say the meeting line, `"Hey assistant, what is the notice period for a general meeting?"`
2. Type `what is our stock price?` — show the refusal
3. Stop, wait for **searchable**, extract action items, point at the quote

Those three show live speech, grounded answers, honest refusal, and verified
extraction — which is the whole system.
