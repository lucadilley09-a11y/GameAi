// Types for GameKit (gamekit.js). The type checker uses these to catch wrong names and wrong arguments in game code
// before it runs (plan: Phase 4, static checkers).

declare namespace GameKit {
  type Mode = 'topdown' | 'platformer';
  type Kind = 'player' | 'enemy' | 'boss' | 'coin' | 'gem' | 'key' | 'door' | 'exit' | 'heart' | 'bullet' | 'item' | string;
  type Key = 'left' | 'right' | 'up' | 'down' | 'action' | 'start' | 'pause';
  type Generator = 'rooms' | 'maze' | 'cave' | 'platforms' | 'open';
  type Direction = [number, number];
  /** A custom enemy brain: returns the direction to move in (it is scaled to the enemy's speed). */
  type Brain = (self: Entity, player: Entity, seconds: number, game: Game) => Direction;
  type BehaviourName = 'still' | 'chase' | 'flee' | 'circle' | 'wander' | 'patrol' | 'zigzag';
  type Event = 'start' | 'tick' | 'collect' | 'hit' | 'defeat' | 'open' | 'reach' | 'jump' | 'respawn' | 'win' | 'lose';
  type HudItem = 'score' | 'lives' | 'time' | 'goal' | ((game: Game) => string);
  type State = 'title' | 'play' | 'paused' | 'win' | 'lose';

  interface GameOptions {
    canvas?: HTMLCanvasElement | null;   // leave out to run headless (tests, playtest bots)
    width?: number; height?: number;     // screen size in pixels (default 640 x 480)
    tile?: number;                       // tile size in pixels (default 32)
    seed?: number;                       // same seed, same game
    mode?: Mode;                         // 'topdown' (default) or 'platformer' (gravity and jumping)
    gravity?: number; jump?: number;     // platformer only
    lives?: number;                      // default 3
    sound?: boolean;                     // default true
    title?: string;                      // shown on the title screen
    background?: string;                 // CSS colour
    pixel?: boolean;                     // draw things as pixel sprites (default true) instead of plain boxes
  }

  interface Entity {
    id: number; kind: Kind; x: number; y: number; w: number; h: number; vx: number; vy: number;
    speed: number; hp: number; maxHp?: number; damage?: number; team: 'player' | 'enemy' | 'none';
    behaviour?: BehaviourName | Brain; axis?: 'x' | 'y';  // patrol direction
    shootEvery?: number;                 // enemies: seconds between shots (0 = never)
    canShoot?: boolean; fireDelay?: number;  // the player: shooting with the action key
    value?: number;                      // coins and gems: points
    pickup?: boolean;                    // any kind: touching it collects it
    fly?: boolean;                       // platformer: not pulled down by gravity
    color: string; dead: boolean; open?: boolean; onGround?: boolean; bumped?: boolean;
    mem: Record<string, any>;            // free storage for brains
  }

  interface Rule { every?: number; when?: Event; n?: number; then: (game: Game, thing?: Entity) => void; }

  interface Game {
    o: Required<GameOptions>; t: number; score: number; lives: number; state: State; why?: string;
    hero: Entity; ents: Entity[]; grid: number[][]; inventory: Record<string, number>; defeated: number; T: number;
    /** Text rows ('#' wall, '.' floor, P player, E enemy, B boss, C coin, G gem, K key, D door, X exit, H heart) or a generator name. */
    level(src: string[] | string | Generator, opts?: { w?: number; h?: number }): Game;
    player(o?: Partial<Entity>): Entity;
    enemy(o?: Partial<Entity> & { boss?: boolean }): Entity;
    add(kind: Kind, o?: Partial<Entity>): Entity;
    /** n of a kind at random open places (enemies start away from the player). */
    spawn(kind: Kind, n?: number, o?: Partial<Entity>): Entity[];
    free(n?: number, away?: { x: number; y: number } | null, minDist?: number): { x: number; y: number }[];
    /** 'collect' a kind (all, or n) | 'reach' the exit | 'survive' n seconds | 'defeat' all enemies (or n). The game is won when every goal is met. */
    goal(type: 'collect' | 'reach' | 'survive' | 'defeat', what?: Kind | null, n?: number): Game;
    rule(r: Rule): Game;
    on(ev: Event, fn: (...a: any[]) => void): Game;
    hud(items: HudItem[]): Game;
    timer(seconds: number): Game;
    sfx(name: 'coin' | 'gem' | 'jump' | 'hit' | 'shoot' | 'key' | 'door' | 'heart' | 'win' | 'lose' | 'powerup'): void;
    shoot(from: Entity, dir: Direction, o?: Partial<Entity>): Entity | null;
    damage(e: Entity, n: number, by?: Entity): void;
    win(why?: string): void; lose(why?: string): void;
    left(kind: Kind): number;
    progress(): number[];
    solidAt(x: number, y: number): boolean;
    start(): Game;
    /** Headless: hold or release a key. */
    press(key: Key, down?: boolean): void;
    /** Headless: advance by seconds (fixed 1/60 s steps); input(game, t) may press keys. Returns the state at the end. */
    run(seconds: number, input?: (game: Game, t: number) => void): State;
    step(dt: number): void;
    log: [number, Event, any][];
  }

  function game(o?: GameOptions): Game;
  function rng(seed: number): (() => number) & { int(n: number): number; pick<T>(a: T[]): T };
  function generate(kind: Generator, w: number, h: number, r: ReturnType<typeof rng>): string[];
  function dirTo(a: { x: number; y: number }, b: { x: number; y: number }): Direction;
  function dist(a: { x: number; y: number }, b: { x: number; y: number }): number;
  const BEHAVIOURS: Record<BehaviourName, Brain>;
  const version: string;
}
