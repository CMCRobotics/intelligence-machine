AFRAME.registerComponent('timed-sound', {
  schema: {
    src: { type: 'selector' }, // Selector for the sound asset
    volume: { type: 'number', default: 1 },
    interval: { type: 'number', default: 5000 }, // Interval in milliseconds
    startOffset: { type: 'number', default: 0 } // Initial delay before first play in milliseconds
  },

  init: function () {
    this.playSound = this.playSound.bind(this);
    this.intervalId = null;
  },

  update: function (oldData) {
    if (this.data.src && this.data.src !== oldData.src) {
      this.stopSoundInterval();
      this.startSoundInterval();
    }
    if (this.data.interval !== oldData.interval) {
      this.stopSoundInterval();
      this.startSoundInterval();
    }
    if (this.data.volume !== oldData.volume) {
      // Update volume if the sound component is already playing
      if (this.el.components.sound) {
        this.el.components.sound.data.volume = this.data.volume;
      }
    }
  },

  play: function () {
    this.startSoundInterval();
  },

  pause: function () {
    this.stopSoundInterval();
  },

  remove: function () {
    this.stopSoundInterval();
  },

  startSoundInterval: function () {
    if (!this.data.src || !this.data.interval) { return; }
    this.stopSoundInterval(); // Clear any existing interval

    // Initial play with startOffset
    if (this.data.startOffset > 0) {
      setTimeout(() => {
        this.playSound();
        this.intervalId = setInterval(this.playSound, this.data.interval);
      }, this.data.startOffset);
    } else {
      this.playSound(); // Play immediately if no offset
      this.intervalId = setInterval(this.playSound, this.data.interval);
    }
  },

  stopSoundInterval: function () {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  },

  playSound: function () {
    const soundComponent = this.el.components.sound;
    if (soundComponent) {
      // Ensure the sound component's volume is updated before playing
      soundComponent.data.volume = this.data.volume;
      soundComponent.playSound();
    } else {
      console.warn('Timed sound: Sound component not found on entity or not ready for:', this.data.src);
    }
  }
});
