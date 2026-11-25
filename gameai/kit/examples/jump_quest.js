// Jump Quest: a platformer. Pick up the key, get past the guard (jump on it!), open the door, reach the exit.
function build(canvas, seed = 1) {
  const g = GameKit.game({ canvas, seed, mode: 'platformer', title: 'Jump Quest', width: 960 });
  g.level([
    '##############################',
    '#............................#',
    '#..........C.................#',
    '#........######......#########',
    '#....................#.......#',
    '#P....K.......E......D....X..#',
    '##############################',
  ]);
  g.ents.filter(e => e.team === 'enemy').forEach(e => { e.behaviour = 'patrol'; e.speed = 60; });
  g.goal('reach', 'exit');
  return g;
}
