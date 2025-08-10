AFRAME.registerComponent('start-experience', {
    init: function() {
        const log = window.log.getLogger('start-experience');
        log.setLevel('debug');
        
        this.el.addEventListener('enter-vr', () => {
            log.debug('Entered VR/AR mode');
            const overlay = document.getElementById('startOverlay');
            if (overlay) overlay.classList.add('hidden');

            // Reset scene position when entering VR
            const scene = document.querySelector('a-scene');
            const worldRoot = document.querySelector('#world-root');
            if (worldRoot) {
                // Position the world slightly in front of the user
                worldRoot.setAttribute('position', '0 0 -2');
                log.debug('Reset world position');
            }
        });
    }
});

AFRAME.registerComponent('world-root', {
    init: function() {
        const log = window.log.getLogger('world-root');
        log.setLevel('debug');
        
        // Initialize position
        this.el.setAttribute('position', '0 0 -1');
        log.debug('World root initialized');
    }
});
