AFRAME.registerComponent('floating-in-jar', {
    schema: {
        bobbingSpeed: {type: 'number', default: 1.0},
        bobbingRange: {type: 'number', default: 0.05}, // meters
        rockingSpeed: {type: 'number', default: 0.5},
        rockingRange: {type: 'number', default: 5}, // degrees
        initialRotation: {type: 'vec3', default: {x: 0, y: 0, z: 0}}
    },

    init: function () {
        this.time = 0;
        this.initialPosition = this.el.object3D.position.clone();
        this.initialRotation = this.el.object3D.rotation.clone();
    },

    tick: function (time, deltaTime) {
        const data = this.data;
        this.time += deltaTime / 1000; // Convert to seconds

        // Bobbing motion (Y-axis)
        const bobbingOffset = Math.sin(this.time * data.bobbingSpeed) * data.bobbingRange;
        this.el.object3D.position.y = this.initialPosition.y + bobbingOffset;

        // Rocking motion (X and Z axes for rotation)
        const rockingX = Math.sin(this.time * data.rockingSpeed * 0.8) * data.rockingRange;
        const rockingZ = Math.cos(this.time * data.rockingSpeed * 1.2) * data.rockingRange;

        this.el.object3D.rotation.x = THREE.MathUtils.degToRad(this.initialRotation.x + rockingX);
        this.el.object3D.rotation.z = THREE.MathUtils.degToRad(this.initialRotation.z + rockingZ);
    }
});
