# Demo speech

The narration only, in order, to read straight through. Actions are marked
in the margin in italics — everything else is spoken.

---

This is a meeting assistant. It listens to a meeting, writes down what is
said, answers questions while the meeting is still going, and pulls out the
action items people committed to.

The part I care about is that all of it runs on this laptop. The speech
recognition, the search, and the language model are all local. There is no
API key anywhere in this project, and no audio or document ever leaves the
machine — which, for a tool that records meetings, is a compliance argument
and not just a cost saving.

Let me show you.

*(start recording)*

So the next item is the notice period for general meetings. I think we need
to check what the policy actually says before we send anything out.

That is appearing as I speak. And you'll notice it arrives in bursts rather
than continuously — it waits until I stop talking before it transcribes.
That's deliberate. Cutting audio on a stopwatch slices words in half.
Cutting at natural pauses gives you clean sentences, and it measurably
improved transcription quality when I changed it.

Now watch this.

Hey assistant — what is the notice period for a general meeting?

*(pause)*

I didn't touch the keyboard. It heard its name, worked out that the rest of
the sentence was the question, searched the company documents, and it's
answering now. And transcription hasn't stopped while it does it — the
meeting doesn't pause for the assistant.

And it cites what it used. The teal chip is a written document, the amber
ones are previous meetings, and the number is how close the match was. So
that's coming from our governance policy — fourteen to sixty days.

Let me try a harder one.

*(type: what does the policy say about volunteers?)*

This is more interesting, because the answer lives in two places. The
written policy says what we're supposed to do about volunteers, and a
previous meeting has people actually discussing it.

Both come back in one ranked list — the document and the meeting,
interleaved. That only works because they're in the same search index. If
I'd kept them in two separate indexes the scores wouldn't be comparable, and
I'd have had to invent some rule for merging them.

And look at the meeting citation. It's a timestamp. I can open the recording
and go listen to that exact moment. A citation you can't check is
decoration.

But the thing I actually care most about is what happens when the answer
isn't there.

*(type: what is our stock price?)*

Nothing. No sources, and it says so.

That's not the model being modest. Nothing scored above the relevance
threshold, so the language model was never called at all. It can't invent an
answer to a question it was never asked to write. That's a rule in code, not
an instruction in a prompt — and a prompt is a request, not a guarantee.

Right — we agreed that Karen will circulate the revised budget to the
members before Friday.

*(pause, then stop recording)*

Watch the status line. The transcript was being saved the whole time, but
saved and searchable are two different things. It's grouping what I just
said into passages, embedding them, and adding them to the index. Now it
says searchable — and only now can I ask about it.

*(type: what did we agree about the budget?)*

And there it is, cited back to my own meeting, seconds after I said it.

*(open Action items, click Extract)*

This last part is the one I had to be most careful about. Asking a model to
find action items works fine. The problem is that it works just as well when
there aren't any — meetings are full of things that sound like commitments
and aren't.

So every task has to come with the sentence it came from, copied exactly,
and the code checks that sentence really is in the transcript before it
stores anything. A model that invents a task has to invent the quote too,
and an invented quote isn't there to find.

There — Karen, circulate the budget, before Friday. And underneath, the
exact words I said, with the timestamp. I can check every line of this
against the recording.

I measured that, by the way. On a meeting that contained no action items at
all, a naive prompt returned thirty-three of them. This returned zero.

So: speech in, transcript stored, questions answered out loud from both
documents and past meetings with citations you can follow, and action items
that are checked rather than trusted.

All of it on one laptop, with no API key and nothing leaving the machine.
About six thousand lines of Python and over a hundred and twenty tests — and
the interesting parts weren't the AI calls. They were the boundaries around
them: deciding where to cut the audio, when to refuse to answer, and how to
verify what the model hands back.

---

# Short version (~90 seconds)

Keeps the four beats that matter: it hears you, it answers from your own
material, it refuses when it should, and it checks its own work.

---

This is a meeting assistant. It listens, answers questions while the meeting
is still running, and pulls out the action items — and all of it runs on
this laptop. No API key, and nothing leaves the machine.

*(start recording)*

So the next item is the notice period for general meetings.

Hey assistant — what is the notice period for a general meeting?

*(pause)*

I didn't touch the keyboard. It heard its name, searched our documents, and
it's answering now — while transcription keeps running underneath. Fourteen
to sixty days, and it cites the policy it took that from.

Now the part I care about more.

*(type: what is our stock price?)*

Nothing, and it says so. That's not modesty — nothing scored above the
relevance threshold, so the model was never called at all. It can't invent
an answer it was never asked to write.

One more thing before I stop. We agreed that Karen will circulate the budget
before Friday.

*(pause, stop recording, open Action items, click Extract)*

There — Karen, circulate the budget, before Friday. And underneath, the
exact sentence I said. Every task has to quote the transcript, and the code
checks that quote is really there before storing it. A model that invents a
task has to invent the quote too.

Speech in, grounded answers out, and nothing kept that can't be checked.
