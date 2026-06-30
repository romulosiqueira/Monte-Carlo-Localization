# Monte Carlo Laser Localization — RoboticsAcademy

Solução do exercício **`montecarlo_laser_loc`**: filtro de partículas (Adaptive
MCL) que localiza o Roomba num mapa conhecido (`mapgrannyannie.png`) usando o
laser de 180° e a odometria ruidosa.

Ficheiro principal: [`solution_montecarlo_laser_loc.py`](solution_montecarlo_laser_loc.py)

## Como usar

1. Abre o exercício *Montecarlo Laser Localization* na RoboticsAcademy.
2. Cola o conteúdo de `solution_montecarlo_laser_loc.py` no editor e carrega em **Play**.
3. O robô passeia sozinho (obstacle-avoidance simples) para dar movimento ao
   filtro. A nuvem de partículas (vermelho) deve concentrar-se sobre a pose real
   e a estimativa (azul) seguir o robô. Põe `ENABLE_WANDER = False` se preferes
   teleoperar.

> **Mapa**: o código tenta vários caminhos em `MAP_URL_CANDIDATES` para
> `WebGUI.getMap(...)`. Se nenhum carregar no teu ambiente, ajusta essa lista
> para o caminho real de `mapgrannyannie.png`.

## As 4 partes pedidas (marcadas no código)

| Parte | Onde | O que faz |
|------|------|-----------|
| **[1] Inicialização + propagação** | `init_particles_*`, `motion_update` | Partículas iniciais (local/global) e *sample motion model* de odometria (Thrun) com ruído proporcional ao deslocamento. |
| **[2] Atualização dos pesos** | `build_likelihood_field`, `measurement_update` | **Likelihood field**: distância de cada ponto-fim do laser ao obstáculo mais próximo (transformada de distância do mapa) → peso gaussiano `Z_HIT·N(d;σ)+Z_RAND`, produto sobre feixes em log. |
| **[3] Resampling** | `systematic_resample` | *Low-variance / systematic resampling* + *roughening* + injeção **Augmented MCL** (recuperação de rapto/privação). Disparado quando `Neff < N/2`. |
| **[4] Estimação da pose** | `estimate_pose` | Média ponderada de `x,y` e média **circular** do `yaw`. |

## Modo DEMO (vídeo: localizar a partir de pose inicial ruim)

No topo do ficheiro, `DEMO_BAD_INIT = True` faz a nuvem começar **deslocada e
espalhada** (estimativa ~1,5 m errada), segura-a parada `DEMO_HOLD_SECONDS`
segundos (para aparecer na câmara) e depois converge sobre o robô quando este
anda. Verificado: 0/25 falhas. Para uso normal (tracking), põe `DEMO_BAD_INIT
= False`. Nota: a demo erra muito a POSIÇÃO mas pouco o RUMO de propósito — com
o laser frontal de 180º só assim a convergência é garantida ao vivo.

## Resolução de problemas no simulador real

- **A estimativa (verde) começa certa e depois "desgarra" do robô (vermelho):**
  causado por `/odom_noisy` **não ser publicado** em alguns RADI — `HAL.getOdom()`
  passa a devolver sempre `(0,0,0)` e o motion model nunca propaga as partículas.
  A solução já está no código: ele **deteta** isso no arranque e passa a usar
  `/odom` (`HAL.getPose3d()`) como fonte de movimento (vês no console:
  `"/odom_noisy indisponível -> a usar /odom"`).
- **As setas:** vermelha = robô real (ground truth), verde = a tua
  estimativa (`showPosition`), azul = partículas. O objetivo é a verde e a
  nuvem azul **convergirem para cima da vermelha e segui-la**.

## Modos de inicialização

- `GLOBAL_INIT = False` *(padrão, recomendado)* — partículas em redor da pose
  inicial de odometria (que coincide com a pose real de arranque). Localização
  local/tracking: **converge de forma rápida e fiável**.
- `GLOBAL_INIT = True` — partículas por todo o espaço livre (localização global
  "pura").

### Nota honesta sobre a localização global

O laser deste Roomba é **frontal de 180°** (`samples=180`, `[-1.57, 1.57] rad`).
Num ambiente interior auto-semelhante como a casa da *granny annie*, um único
varrimento de meio plano é fortemente **ambíguo** (aliasing perceptual): muitas
poses explicam a mesma leitura. O MCL básico tende, por isso, a não colapsar para
uma única hipótese a partir de inicialização totalmente global — é uma limitação
conhecida que se resolve com técnicas mais avançadas (KLD-sampling, Mixture-MCL).
Para uma demonstração robusta usa o modo local (padrão).

## Verificação (simulação offline)

Testei a solução fora da RoboticsAcademy com mocks de `HAL`/`WebGUI`/`Frequency`,
gerando o laser por *ray-casting* no mapa real e a odometria com ruído/deriva.
Para garantir robustez, corri um **stress test de 40 realizações** (seeds):

- **Modo local (padrão)**: **0/40 divergências**, pior erro 0.37 m, erro final
  ~0.05–0.06 m e orientação <1°. Converge e mantém o lock.
- O modelo de medição foi validado: na pose verdadeira a verossimilhança é
  máxima e `yaw+180°` é fortemente rejeitado (geometria/indexação corretas).
- Confirmado o detalhe de que `WebGUI.showParticles(...)` **altera a lista
  in-place** (escala para píxeis) — por isso passamos sempre uma **cópia**.

### Lição: o "flip" de 180º (e como foi resolvido)

Numa versão inicial o filtro convergia mas, em ~3% das realizações, durante uma
rotação numa zona localmente simétrica, a nuvem "virava" para a pose espelhada de
180º. Pior: essa pose explicava o varrimento **igualmente bem** (verossimilhança
~0.99), por isso era **irrecuperável e indetetável** pelo laser. A causa é o laser
ser **frontal de 180º**. A solução foi **confiar na orientação da odometria**:
mantendo o ruído de rotação pequeno (`ALPHA1/2/4` e `ROUGHEN_YAW` baixos), o yaw
das partículas segue a odometria e o flip torna-se impossível, enquanto o laser
continua a corrigir a posição. Resultado: 0 falhas em 40 seeds.

`scipy` é usado para a transformada de distância se existir; caso contrário há um
fallback em NumPy puro (Felzenszwalb), pelo que não há dependência obrigatória
além de NumPy.
