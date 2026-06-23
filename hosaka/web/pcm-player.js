// AudioWorklet that plays a queue of float32 PCM chunks (24 kHz mono) pushed
// from the main thread over the port. Underruns emit silence rather than
// glitching, so a slow stream just gaps instead of crashing playback.
class PCMPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this._queue = []; // pending Float32Array chunks
    this._cur = null; // chunk currently being drained
    this._pos = 0;
    this.port.onmessage = (e) => {
      if (e.data === null) {
        // flush (e.g. on a new utterance / cancel)
        this._queue.length = 0;
        this._cur = null;
        this._pos = 0;
        return;
      }
      this._queue.push(e.data);
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    if (!out) return true;
    for (let i = 0; i < out.length; i++) {
      if (!this._cur || this._pos >= this._cur.length) {
        this._cur = this._queue.shift() || null;
        this._pos = 0;
      }
      out[i] = this._cur ? this._cur[this._pos++] : 0;
    }
    return true; // keep the processor alive across utterances
  }
}

registerProcessor("pcm-player", PCMPlayer);
