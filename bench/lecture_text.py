"""Varied, lecture-like English text for the synthetic benchmark media.

A seeded generator walks through topic *blocks* (a few paragraphs on one subject, like a
stretch of a lecture). Each paragraph mixes hand-written topic facts with discourse
templates ("Notice how closely {term} and {term2} are linked.") filled from the topic's
own vocabulary, so a 2.5-hour file is thousands of distinct sentences, not one paragraph
on repeat. Students occasionally ask a question in another voice.

The text avoids things that make WER measure formatting instead of recognition: no digits
or number words, no abbreviations, no hyphenated compounds, and (mostly) no words with
different US and UK spellings.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Topic:
    name: str  # used after "on", "about": "the water cycle"
    field: str  # "earth science"
    terms: tuple[str, ...]  # noun phrases that need no article added
    scenes: tuple[str, ...]  # clauses that follow "when" / "imagine that"
    facts: tuple[str, ...]  # complete sentences


@dataclass
class Paragraph:
    text: str
    topic: str
    kind: str  # "lecture" or "question"


@dataclass
class Block:
    topic: str
    paragraphs: list[Paragraph] = field(default_factory=list)


TOPICS: tuple[Topic, ...] = (
    Topic(
        "the water cycle", "earth science",
        ("evaporation", "condensation", "precipitation", "groundwater", "surface runoff",
         "water vapor", "cloud formation", "the dew point", "transpiration from plants",
         "the heat stored in the oceans"),
        ("a puddle slowly disappears on a warm afternoon",
         "a cold glass of lemonade gathers drops of water on the outside",
         "rain falls on a hillside and trickles toward a stream",
         "fog settles over a lake early in the morning",
         "snow on a mountain melts during a sunny spring week"),
        ("Water is constantly moving between the oceans, the air, and the land.",
         "The sun provides the energy that lifts water from the surface into the atmosphere.",
         "Warm air can hold far more moisture than cold air, which is why clouds form as air rises and cools.",
         "Much of the rain that falls on land eventually finds its way back to the sea through rivers.",
         "Some water soaks deep into the ground and may stay there for an extremely long time.",
         "Plants pull water up from their roots and release it through tiny openings in their leaves.",
         "Without this cycle, fresh water would quickly run out in most parts of the world."),
    ),
    Topic(
        "cell biology", "biology",
        ("the cell membrane", "the nucleus", "mitochondria", "ribosomes", "the cytoplasm",
         "protein synthesis", "cell division", "the endoplasmic reticulum", "enzymes",
         "cellular respiration"),
        ("a single bacterium splits in half to make a pair of new cells",
         "a muscle cell burns sugar during a long run",
         "a white blood cell swallows an invading germ",
         "a seed begins to grow after the first spring rain",
         "a cut on your finger slowly heals over a week"),
        ("Every living thing is made of cells, and every cell comes from an earlier cell.",
         "The membrane decides what enters and leaves the cell, a bit like a careful security guard.",
         "Mitochondria are often called the power plants of the cell because they release energy from food.",
         "The nucleus stores genetic instructions that tell the cell which proteins to build.",
         "Ribosomes read those instructions and link small building blocks into long protein chains.",
         "Plant cells have a rigid wall around them, which animal cells do not have.",
         "When cells divide without control, the result can be a dangerous growth."),
    ),
    Topic(
        "plate tectonics", "geology",
        ("the earth's crust", "the mantle", "tectonic plates", "subduction zones",
         "continental drift", "mid ocean ridges", "fault lines", "volcanic activity",
         "the spreading of the ocean floor", "mountain building"),
        ("an earthquake shakes a coastal city in the middle of the night",
         "a volcano on a remote island begins to rumble",
         "hikers find fossil shells near the top of a mountain",
         "a road slowly cracks along a fault over many years",
         "maps of the coastlines of Africa and South America are laid side by side"),
        ("The outer shell of the earth is broken into large plates that move very slowly.",
         "Plates move about as fast as your fingernails grow.",
         "Where plates pull apart, molten rock rises and creates new ocean floor.",
         "Where plates collide, one may sink beneath the other, or both may crumple upward into mountains.",
         "Most earthquakes and volcanoes are found along the edges of plates.",
         "The idea of drifting continents was mocked for decades before the evidence became overwhelming.",
         "The highest mountains on the planet were pushed up by the collision of India with the rest of Asia."),
    ),
    Topic(
        "supply and demand", "economics",
        ("market prices", "consumer demand", "the supply curve", "price elasticity", "scarcity",
         "market equilibrium", "a price ceiling", "competition between sellers",
         "production costs", "consumer preferences"),
        ("a sudden frost destroys much of the orange harvest",
         "a popular toy sells out weeks before the holidays",
         "a new factory opens and floods the market with cheap shoes",
         "the price of gasoline jumps overnight",
         "a small town gets its first coffee shop"),
        ("When a product becomes scarce, buyers compete for it and the price tends to rise.",
         "When sellers produce more than people want, they usually have to lower their prices.",
         "The point where the amount supplied matches the amount demanded is called equilibrium.",
         "Some goods, like salt or medicine, barely change in demand even when prices change.",
         "Luxury goods often see demand fall sharply when prices go up.",
         "Governments sometimes set price limits, which can lead to shortages or long lines.",
         "Prices carry information, telling producers where their effort is most needed."),
    ),
    Topic(
        "the French Revolution", "history",
        ("the monarchy", "the Estates General", "the storming of the Bastille",
         "the Reign of Terror", "the Declaration of the Rights of Man", "the rise of Napoleon",
         "royal debt", "the National Assembly", "popular uprisings", "the guillotine"),
        ("crowds gather outside a fortress in Paris on a hot summer day",
         "a king tries to escape the capital in disguise",
         "bread prices double in a single winter",
         "a newspaper publishes an angry attack on the queen",
         "delegates refuse to leave a tennis court until they write a constitution"),
        ("By the end of the old regime, the French crown was deeply in debt.",
         "Ordinary people paid most of the taxes, while nobles and clergy paid very little.",
         "The revolution began with demands for reform and ended up overturning the entire social order.",
         "Ideas about liberty and equality spread quickly through pamphlets and public debates.",
         "The king and queen were eventually tried and executed.",
         "During the Terror, a great many suspected enemies of the revolution were put to death.",
         "In the end, a young general named Napoleon seized power and crowned himself emperor."),
    ),
    Topic(
        "Newton's laws of motion", "physics",
        ("inertia", "force", "acceleration", "mass", "momentum", "friction", "gravity",
         "action and reaction", "net force", "free fall"),
        ("a passenger lurches forward when a bus stops suddenly",
         "an astronaut pushes off from the wall of a space station",
         "a hockey puck slides across smooth ice",
         "a child jumps off a small boat onto a dock",
         "a heavy shopping cart is harder to push than an empty one"),
        ("An object at rest stays at rest unless something pushes or pulls on it.",
         "A moving object keeps moving in a straight line unless a force changes its path.",
         "The same push will speed up a light object much more than a heavy one.",
         "Every time you push on something, it pushes back on you just as hard.",
         "Rockets work by throwing gas backward, which pushes the rocket forward.",
         "Friction is the reason a rolling ball eventually slows down and stops.",
         "Newton saw that the force pulling an apple down also keeps the moon in orbit."),
    ),
    Topic(
        "photosynthesis", "plant biology",
        ("chlorophyll", "sunlight", "carbon dioxide", "glucose", "the light reactions",
         "the Calvin cycle", "chloroplasts", "oxygen release", "leaf structure", "stomata"),
        ("a potted plant leans toward a sunny window",
         "a forest turns bright green in early spring",
         "a pond fills with algae during a hot summer",
         "leaves turn yellow and red as autumn arrives",
         "a seedling grows in a dark closet and turns pale"),
        ("Plants capture energy from sunlight and store it in the chemical bonds of sugar.",
         "They take in carbon dioxide from the air and release oxygen along the way.",
         "Chlorophyll absorbs red and blue light and reflects green, which is why leaves look green.",
         "Nearly all the food energy on earth can be traced back to photosynthesis.",
         "The oxygen we breathe was produced by plants and tiny ocean organisms over enormous spans of time.",
         "When light is weak or water is scarce, photosynthesis slows down.",
         "Some desert plants open their pores only at night to save water."),
    ),
    Topic(
        "memory and learning", "psychology",
        ("short term memory", "long term memory", "working memory", "attention", "rehearsal",
         "the forgetting curve", "spaced repetition", "retrieval practice", "sleep and memory",
         "emotional memories"),
        ("you forget a phone number moments after hearing it",
         "a student crams the night before an exam and remembers little a week later",
         "a familiar song brings back a summer from your childhood",
         "you walk into a room and forget why you came in",
         "a chess master glances at a board and remembers every piece"),
        ("Memory is not a recording; it is rebuilt every time we recall it.",
         "We forget most new information quickly unless we return to it.",
         "Spreading study sessions over several days works far better than cramming.",
         "Testing yourself is one of the most effective ways to make knowledge stick.",
         "Sleep helps the brain sort and strengthen the memories formed during the day.",
         "Strong emotions can make a memory vivid, but not necessarily accurate.",
         "Working memory can only hold a handful of items at once."),
    ),
    Topic(
        "the Roman Empire", "ancient history",
        ("the Roman Senate", "the legions", "Roman roads", "the aqueducts", "the Pax Romana",
         "provincial government", "Latin literature", "Roman law", "the emperor's power",
         "the fall of the western empire"),
        ("a merchant travels by road from Spain to the city of Rome",
         "soldiers build a wall across northern Britain",
         "crowds fill the Colosseum for a day of games",
         "a farmer in Gaul pays taxes to a Roman official",
         "water flows into a city through stone channels from distant hills"),
        ("Rome grew from a small city into an empire that surrounded the entire Mediterranean Sea.",
         "Its roads were so well built that some are still used today.",
         "Roman law shaped the legal systems of many modern countries.",
         "Latin, the language of Rome, gave rise to French, Spanish, Italian, and Portuguese.",
         "For a long stretch, the empire enjoyed relative peace and growing trade.",
         "Historians still debate why the western half of the empire collapsed.",
         "The eastern half survived for many more centuries, with its capital at Constantinople."),
    ),
    Topic(
        "probability", "mathematics",
        ("random events", "independent events", "the law of large numbers", "expected value",
         "conditional probability", "sample spaces", "the gambler's fallacy",
         "probability distributions", "statistical noise", "base rates"),
        ("you flip a fair coin and get heads several times in a row",
         "a weather forecaster says there is a good chance of rain tomorrow",
         "a doctor interprets a positive result from a medical test",
         "a casino offers a game that looks generous at first glance",
         "you draw a card from a shuffled deck and try to guess its suit"),
        ("A fair coin has no memory, so past flips do not change the next one.",
         "Over many trials, the share of heads gets closer and closer to one half.",
         "Expected value tells us the average outcome if we could repeat a bet many times.",
         "Rare events happen surprisingly often when there are many chances for them to occur.",
         "A test can be quite accurate and still produce mostly false alarms when a disease is rare.",
         "Our intuition about chance is often wrong, which is why careful calculation matters.",
         "Probability began as a way to settle disputes about games of chance."),
    ),
    Topic(
        "electricity and circuits", "physics",
        ("electric current", "voltage", "resistance", "series circuits", "parallel circuits",
         "electric charge", "conductors and insulators", "the battery", "short circuits",
         "electrical power"),
        ("a string of holiday lights goes dark when a single bulb fails",
         "you rub a balloon on your hair and it sticks to the wall",
         "a phone charger warms up while it is plugged in",
         "a lightning bolt strikes a tall tree",
         "a flashlight grows dim as its battery runs low"),
        ("Electric current is the flow of charge through a material.",
         "Voltage acts like pressure that pushes charge around a circuit.",
         "Resistance limits how much current can flow for a given voltage.",
         "Metals conduct electricity well because their electrons can move freely.",
         "In a parallel circuit, each device gets its own path, so one failure does not stop the others.",
         "A short circuit lets current bypass the load, which can cause dangerous heating.",
         "Power tells us how quickly electrical energy is being used."),
    ),
    Topic(
        "ecosystems and food webs", "ecology",
        ("producers", "consumers", "decomposers", "the food web",
         "predator and prey relationships", "keystone species", "energy flow", "biodiversity",
         "habitat loss", "invasive species"),
        ("wolves return to a national park after a long absence",
         "a lake turns green after runoff from nearby farms washes in",
         "a fallen log slowly rots on the forest floor",
         "a new beetle arrives on a ship and spreads through local trees",
         "rabbits multiply quickly on an island with no foxes"),
        ("Energy enters most ecosystems through plants and flows upward through the animals that eat them.",
         "Only a small part of the energy at each level reaches the next level.",
         "Decomposers break down dead material and return nutrients to the soil.",
         "Removing a single species can set off a chain of changes throughout an ecosystem.",
         "Ecosystems with many species tend to recover more easily from shocks.",
         "Invasive species often thrive because their natural enemies were left behind.",
         "Healthy wetlands filter water and protect coastlines from storms."),
    ),
    Topic(
        "the immune system", "medicine",
        ("white blood cells", "antibodies", "vaccines", "inflammation", "the lymph nodes",
         "immune memory", "fever", "autoimmune disease", "pathogens", "herd immunity"),
        ("a small cut on your knee turns red and warm",
         "you catch a cold from a classmate",
         "a child receives a routine vaccine at the doctor's office",
         "your glands feel swollen during a sore throat",
         "a whole school stays healthy during flu season because most students were vaccinated"),
        ("The immune system protects the body from germs, toxins, and damaged cells.",
         "White blood cells patrol the body looking for anything that does not belong.",
         "Antibodies attach to invaders and mark them for destruction.",
         "After an infection, the body remembers the germ and can respond faster next time.",
         "Vaccines train the immune system without making us seriously ill.",
         "A fever makes it harder for many germs to multiply.",
         "Sometimes the immune system mistakenly attacks the body's own tissues."),
    ),
    Topic(
        "computer algorithms", "computer science",
        ("sorting algorithms", "binary search", "recursion", "running time", "data structures",
         "hash tables", "graph search", "dynamic programming", "the worst case",
         "memory usage"),
        ("you look up a word in a paper dictionary by opening it near the middle",
         "a delivery company plans routes for a large fleet of trucks",
         "a search engine returns results in a fraction of a second",
         "a librarian sorts a pile of returned books by author",
         "a navigation app finds the fastest way across a busy city"),
        ("An algorithm is simply a precise list of steps for solving a problem.",
         "Some methods slow down only a little as the input grows, while others become hopeless.",
         "Binary search works by cutting the search space in half at every step.",
         "Choosing the right data structure can matter more than clever code.",
         "Recursion solves a problem by breaking it into smaller copies of itself.",
         "Many important problems have no known method that is both fast and exact.",
         "Good programmers think about the worst case, not just the typical case."),
    ),
    Topic(
        "climate and weather", "atmospheric science",
        ("air pressure", "the jet stream", "ocean currents", "the greenhouse effect", "humidity",
         "weather fronts", "seasonal patterns", "global temperatures", "trade winds",
         "polar ice"),
        ("a cold front sweeps through and the temperature drops sharply",
         "a hurricane gathers strength over warm ocean water",
         "a mountain town gets heavy snow while the valley stays dry",
         "a coastal city stays mild through the winter",
         "a spell of very hot weather settles over a large region for several weeks"),
        ("Weather describes conditions over hours and days, while climate describes patterns over decades.",
         "The greenhouse effect keeps the planet warm enough for life.",
         "Adding more heat trapping gases strengthens that effect and warms the planet.",
         "Ocean currents carry warm water toward the poles and cold water back toward the equator.",
         "The tilt of the earth, not its distance from the sun, causes the seasons.",
         "Mountains force air upward, which often makes one side wet and the other side dry.",
         "Melting ice reflects less sunlight, which leads to even more warming."),
    ),
    Topic(
        "the Renaissance", "art history",
        ("linear perspective", "humanism", "patronage", "fresco painting",
         "classical sculpture", "the printing press", "oil painting techniques",
         "the Medici family", "Renaissance architecture", "anatomical drawing"),
        ("a young painter studies ancient statues in the ruins of Rome",
         "a wealthy banker pays an artist to decorate a chapel",
         "a printer produces copies of a book far faster than any scribe",
         "an artist studies a body to understand how muscles move",
         "visitors look up at a ceiling covered with painted figures"),
        ("The Renaissance began in Italian cities and spread across Europe over the following centuries.",
         "Artists rediscovered the art and writing of ancient Greece and Rome.",
         "Painters learned to create the illusion of depth on a flat surface.",
         "Wealthy families competed to sponsor the most impressive buildings and paintings.",
         "Leonardo da Vinci filled notebooks with sketches of machines, plants, and the human body.",
         "The printing press made books cheaper and helped new ideas travel quickly.",
         "Michelangelo spent years painting the ceiling of the Sistine Chapel."),
    ),
    Topic(
        "chemical reactions", "chemistry",
        ("reactants and products", "chemical bonds", "activation energy", "catalysts",
         "reaction rates", "oxidation", "acids and bases", "the conservation of mass",
         "chemical equilibrium", "exothermic reactions"),
        ("an iron gate slowly rusts in the rain",
         "baking soda fizzes when you pour vinegar on it",
         "a campfire crackles and turns wood into ash and smoke",
         "a sliced apple turns brown on the kitchen counter",
         "a hand warmer heats up when you shake it"),
        ("In a chemical reaction, atoms are rearranged, but none are created or destroyed.",
         "Breaking bonds takes energy, and forming new bonds releases energy.",
         "Heating a mixture usually makes a reaction go faster.",
         "A catalyst speeds up a reaction without being used up.",
         "Rust forms when iron combines with oxygen in the presence of water.",
         "Some reactions give off heat, while others absorb it from their surroundings.",
         "Many reactions can run in both directions until they reach a balance."),
    ),
    Topic(
        "language acquisition", "linguistics",
        ("the critical period", "grammar rules", "vocabulary growth", "babbling",
         "bilingual children", "sign languages", "irregular verbs", "first words",
         "sentence structure", "accent and pronunciation"),
        ("a toddler points at a dog and shouts its name with delight",
         "a family moves abroad and the children pick up the new language within months",
         "an adult learner struggles with sounds that do not exist in their first language",
         "a baby turns toward the sound of its mother's voice",
         "children on a playground invent their own secret words"),
        ("Children learn language remarkably fast without formal lessons.",
         "Babies begin by telling apart the sounds of every language, then tune in to the ones they hear.",
         "Young children often apply grammar rules too broadly before learning the exceptions.",
         "Learning a second language early makes a native accent much more likely.",
         "Sign languages have full grammars, just like spoken languages.",
         "Children raised with two languages may mix them, but they keep them apart over time.",
         "Talking with children, not just around them, helps their vocabulary grow."),
    ),
    Topic(
        "astronomy and the solar system", "astronomy",
        ("planetary orbits", "the asteroid belt", "the gas giants", "the phases of the moon",
         "solar eclipses", "comets", "the speed of light", "the habitable zone",
         "the core of the sun", "the rocky planets"),
        ("the moon passes between the earth and the sun and the sky turns dark at midday",
         "a bright comet appears with a long glowing tail",
         "you look up on a clear night far from city lights",
         "a space probe sends back pictures of the rings of Saturn",
         "a meteor streaks across the sky during a summer night"),
        ("The planets all orbit the sun in nearly the same flat plane.",
         "The inner planets are small and rocky, while the outer planets are huge balls of gas and ice.",
         "The sun produces energy by fusing hydrogen into helium deep in its core.",
         "Light from the sun takes several minutes to reach the earth.",
         "The moon does not make its own light; it reflects light from the sun.",
         "Comets are frozen leftovers from the early days of the solar system.",
         "Galileo's telescope revealed moons circling Jupiter, which challenged older ideas about the heavens."),
    ),
    Topic(
        "ethics and moral philosophy", "philosophy",
        ("moral duty", "the consequences of our actions", "virtue", "fairness",
         "personal freedom", "moral responsibility", "the greater good", "human rights",
         "moral dilemmas", "character"),
        ("you find a wallet full of cash on an empty street",
         "a doctor must decide which patient receives a scarce treatment",
         "a friend asks you to keep a secret that could hurt someone else",
         "a company discovers a safety problem in a product it already sold",
         "a runaway trolley heads toward a group of workers on the track"),
        ("Some philosophers judge actions by their results, while others focus on duties and rules.",
         "Aristotle argued that a good life comes from developing good habits of character.",
         "Kant believed we should never treat people merely as tools for our own goals.",
         "Many moral debates come down to conflicts between freedom and fairness.",
         "Thought experiments help us test our moral intuitions in a controlled way.",
         "Reasonable people can disagree deeply about difficult moral questions.",
         "Ethics is not only about big dilemmas; it shapes small daily choices as well."),
    ),
    Topic(
        "the circulatory system", "human physiology",
        ("the heart", "arteries", "veins", "capillaries", "blood pressure", "red blood cells",
         "heart rate", "oxygen delivery", "the circulation through the lungs", "blood clotting"),
        ("your heart pounds after you sprint up a flight of stairs",
         "a nurse wraps a cuff around your arm to check your blood pressure",
         "your fingers turn pale on a freezing winter morning",
         "a small scrape stops bleeding after a few minutes",
         "a climber feels short of breath high in the mountains"),
        ("The heart is a muscle that pumps blood through the body without ever taking a break.",
         "Arteries carry blood away from the heart, and veins carry it back.",
         "Capillaries are so narrow that red blood cells pass through them one at a time.",
         "Blood delivers oxygen and nutrients and carries away waste.",
         "Regular exercise makes the heart stronger and more efficient.",
         "High blood pressure strains the heart and blood vessels over time.",
         "William Harvey showed that blood circulates in a loop rather than being used up."),
    ),
    Topic(
        "the Industrial Revolution", "economic history",
        ("steam power", "the factory system", "textile mills", "the growth of cities",
         "working children", "the railways", "coal mining", "mass production",
         "working conditions", "trade unions"),
        ("a family leaves their farm to find work in a crowded city",
         "a steam engine pumps water out of a flooded mine",
         "a weaver loses her job when a mechanical loom arrives",
         "the first passenger train pulls into a station",
         "smoke from countless chimneys darkens the sky"),
        ("The Industrial Revolution began in Britain and transformed how goods were made.",
         "Machines powered by water and steam replaced much of the work once done by hand.",
         "Cities grew rapidly as people moved closer to factories.",
         "Working days were long, and conditions in many factories were dangerous.",
         "Railways shrank travel times and connected distant markets.",
         "Workers eventually joined together to demand better pay and safer conditions.",
         "The changes brought great wealth, but the benefits were shared very unevenly."),
    ),
    Topic(
        "genetics and heredity", "biology",
        ("genes", "dominant and recessive traits", "DNA", "chromosomes", "mutations",
         "inherited diseases", "genetic variation", "natural selection", "gene expression",
         "family trees"),
        ("a pair of brown eyed parents have a child with blue eyes",
         "a gardener crosses tall pea plants with short ones",
         "a population of moths grows darker as factory soot covers the trees",
         "a family discovers that a rare condition runs through several generations",
         "identical twins grow up in different countries"),
        ("Genes carry the instructions passed from parents to their children.",
         "Gregor Mendel discovered the basic rules of heredity by breeding pea plants.",
         "A recessive trait can skip a generation and then reappear.",
         "Mutations are changes in the genetic code, and most of them are harmless.",
         "Natural selection rewards traits that help organisms survive and reproduce.",
         "The same gene can behave differently depending on the environment.",
         "Darwin developed his ideas without knowing how traits were actually inherited."),
    ),
    Topic(
        "sound and waves", "physics",
        ("frequency", "wavelength", "amplitude", "echoes", "the speed of sound", "resonance",
         "the Doppler effect", "pitch", "sound waves", "vibration"),
        ("an ambulance siren drops in pitch as it passes you",
         "you shout across a canyon and hear your voice return",
         "a singer shatters a glass with a single high note",
         "a guitarist tightens a string and the note rises",
         "thunder rumbles a few seconds after a flash of lightning"),
        ("Sound is a vibration that travels through air, water, or solid materials.",
         "Higher frequencies sound higher in pitch.",
         "Sound cannot travel through empty space, because there is nothing to vibrate.",
         "Sound moves much faster through water than through air.",
         "An echo is simply sound that bounces off a surface and returns to us.",
         "When an object vibrates at its natural frequency, resonance can make the motion grow very large.",
         "The pitch of a moving source changes because the waves bunch up in front of it and spread out behind it."),
    ),
)

OPENERS = (
    "Let's turn now to {name}.",
    "Today I want to spend some time on {name}.",
    "Our next topic comes from {field}, and it concerns {name}.",
    "Before the break, we were talking about {name}, so let's pick up that thread.",
    "If you did the reading for this week, you already know a little about {name}.",
    "Here is a question that comes up every year when we study {name}.",
    "Let's start with a simple picture of {name}.",
    "Now, I know that {name} can feel abstract at first, so let's make it concrete.",
    "Let's go a little deeper into {name}.",
    "There is one more side of {name} that I want to cover.",
    "So, back to {name}.",
    "Let's continue with {name}.",
)

BODY = (
    "The key idea to remember here is {term}.",
    "Notice how closely {term} and {term2} are linked.",
    "A good way to picture {term} is to imagine that {scene}.",
    "Think about what happens when {scene}.",
    "When {scene}, what you are really watching is {term} at work.",
    "Many students mix up {term} and {term2}, but they describe different things.",
    "The difference between {term} and {term2} will matter a lot on the exam.",
    "So if something disturbs {term}, you should expect a change in {term2} as well.",
    "People who work in {field} still argue about the exact role of {term}.",
    "Please write down {term} in your notes, because we will come back to it.",
    "It helps to sketch a small diagram that connects {term} to {term2}.",
    "This is exactly the point where {term} starts to matter.",
    "Keep in mind that it is much easier to describe {term} than to measure it.",
    "Most textbooks introduce {term} before {term2}, but I prefer the opposite order.",
    "For example, when {scene}, the role of {term} becomes much easier to see.",
    "You might fairly ask why anyone should care about {term}.",
    "The honest answer is that we still do not fully understand {term}.",
    "Early thinkers described {term} in very different language from the words we use today.",
    "If you only remember one thing from today, remember how {term} connects to {term2}.",
    "Some of you asked me last time about {term}, and that was a great question.",
    "Look at it from another angle and you will find {term2} hiding behind {term}.",
    "On the exam, I often ask students to compare {term} with {term2}.",
    "Imagine, for a moment, that {scene}.",
    "In that situation, the first thing to look at is {term}.",
    "A common mistake is to think about {term} without thinking about {term2}.",
    "The textbook gives a careful definition of {term}, and it is worth reading slowly.",
    "We can test this idea by asking what would happen without {term}.",
    "The surprising part is how much {term2} depends on {term}.",
    "Let me give you a simple example from ordinary life. Suppose that {scene}.",
    "Now connect that back to {term}, and the whole picture becomes clearer.",
    "That is the reason careful studies of {term} took such a long time.",
    "In the discussion section this week, you will work through a problem about {term}.",
    "Here is a way to remember it: first comes {term}, and then comes {term2}.",
    "Every serious account of {name} has to explain {term}.",
    "Specialists in {field} tend to start with {term} and work outward from there.",
    "So far we have mostly talked about {term}, but {term2} deserves equal attention.",
    "The more closely you study {term}, the stranger it begins to look.",
    "Once you understand {term}, many of the later chapters will feel much easier.",
    "I want you to notice that {scene}, and ask yourself why.",
    "There is a lovely old demonstration of {term} that still works in a classroom.",
)

CLOSERS = (
    "We will build on this idea next week.",
    "Take a moment to check that this makes sense before we move on.",
    "Hold on to that thought, because it will matter later.",
    "Try explaining this to a friend tonight, and you will quickly see where the gaps are.",
    "That is enough on this point for now.",
    "Keep that example in mind as we go forward.",
    "We will see the same pattern again in a different setting.",
    "Make sure this part is clear in your notes before the next class.",
)

QUESTIONS = (
    "Sorry, could you explain again how {term} relates to {term2}?",
    "Is {term} basically the same thing as {term2}, or are they different?",
    "Will we need to know about {term} for the final exam?",
    "Sorry, I missed the part about {term}. Could you go over it once more?",
    "How do we actually know that {term} works the way you described?",
    "Does {term} always come before {term2}, or can it happen the other way around?",
)

ANSWERS = (
    "That is a fair question.",
    "Good question, and I am glad you asked it.",
    "Sure, let me try a different explanation.",
    "Let me answer that carefully.",
    "Great, that gives me a chance to clear something up.",
)

MAX_FACT_USES = 2  # per generated file


class LectureTextGenerator:
    """Yields topic blocks of paragraphs, deterministically for a given seed."""

    def __init__(self, seed: int, *, question_rate: float = 0.08) -> None:
        self._rng = random.Random(seed)
        self._question_rate = question_rate
        self._fact_uses: dict[str, int] = {}
        self._topic_queue: list[Topic] = []
        self._last_topic: Topic | None = None

    def blocks(self) -> Iterator[Block]:
        while True:
            yield self._block()

    def _next_topic(self) -> Topic:
        if not self._topic_queue:
            self._topic_queue = list(TOPICS)
            self._rng.shuffle(self._topic_queue)
            if self._topic_queue[-1] is self._last_topic:  # never the same topic twice in a row
                self._topic_queue.insert(0, self._topic_queue.pop())
        self._last_topic = self._topic_queue.pop()
        return self._last_topic

    def _block(self) -> Block:
        rng, topic = self._rng, self._next_topic()
        block = Block(topic.name)
        count = rng.randint(4, 8)
        for i in range(count):
            sentences: list[str] = []
            if i == 0:
                sentences.append(self._fill(rng.choice(OPENERS), topic))
            elif block.paragraphs and rng.random() < self._question_rate:
                question = self._fill(rng.choice(QUESTIONS), topic)
                block.paragraphs.append(Paragraph(question, topic.name, "question"))
                sentences.append(rng.choice(ANSWERS))
            sentences += self._body(topic, rng.randint(3, 6))
            if i == count - 1 and rng.random() < 0.6:
                sentences.append(rng.choice(CLOSERS))
            block.paragraphs.append(Paragraph(" ".join(sentences), topic.name, "lecture"))
        return block

    def _body(self, topic: Topic, n: int) -> list[str]:
        out: list[str] = []
        recent: set[str] = set()
        while len(out) < n:
            fact = self._fresh_fact(topic) if self._rng.random() < 0.35 else None
            if fact is not None:
                out.append(fact)
                continue
            template = self._rng.choice(BODY)
            if template in recent:
                continue
            recent.add(template)
            out.append(self._fill(template, topic))
        return out

    def _fresh_fact(self, topic: Topic) -> str | None:
        unused = [f for f in topic.facts if self._fact_uses.get(f, 0) < MAX_FACT_USES]
        if not unused:
            return None
        least = min(self._fact_uses.get(f, 0) for f in unused)
        fact = self._rng.choice([f for f in unused if self._fact_uses.get(f, 0) == least])
        self._fact_uses[fact] = self._fact_uses.get(fact, 0) + 1
        return fact

    def _fill(self, template: str, topic: Topic) -> str:
        term, term2 = self._rng.sample(topic.terms, 2)
        text = template.format(
            name=topic.name, field=topic.field, term=term, term2=term2,
            scene=self._rng.choice(topic.scenes),
        )
        return _capitalize_sentences(text)


def _capitalize_sentences(text: str) -> str:
    """Upper-case the first letter of each sentence ("the earth's crust is" -> "The earth's ...")."""
    chars = list(text)
    start = True
    for i, ch in enumerate(chars):
        if start and ch.isalpha():
            chars[i] = ch.upper()
            start = False
        elif ch in ".?!":
            start = True
    return "".join(chars)
