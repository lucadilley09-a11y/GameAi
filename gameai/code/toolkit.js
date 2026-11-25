// The game toolkit: the only helpers an enemy brain can use (besides Math). Same in the game and the tests.
// A brain is the body of: function brain(e, p, t) { ... return [dx, dy]; }
//   e: the enemy {x, y, mem}   p: the player {x, y}   t: seconds since the start
//   returns the direction to move in; the game turns it into the enemy's speed.
function dist(a, b) { return Math.hypot(b.x - a.x, b.y - a.y); }
function dir(a, b) { const d = Math.hypot(b.x - a.x, b.y - a.y) || 1; return [(b.x - a.x) / d, (b.y - a.y) / d]; }
