"""Willowbrook: a small town for rule-based generative agents.

Twenty-five residents with jobs, homes, weekly routines and favourite places.
Everything here is static data; the dynamics live in sim.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STEPS_PER_DAY = 96          # one step = 15 game minutes
STEP_HOURS = 24.0 / STEPS_PER_DAY
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
START_MONTH, START_DAY = "March", 2   # day 0 is Monday, March 2
MONTH_DAYS = [("March", 31), ("April", 30), ("May", 31), ("June", 30), ("July", 31),
              ("August", 31), ("September", 30), ("October", 31), ("November", 30),
              ("December", 31), ("January", 31), ("February", 28)]


def date_str(day: int) -> str:
    d = START_DAY - 1 + day
    for month, n in MONTH_DAYS * 3:
        if d < n:
            return f"{WEEKDAYS[day % 7]}, {month} {d + 1}"
        d -= n
    raise ValueError(day)


def weekday(day: int) -> str:
    return WEEKDAYS[day % 7]


PLACES = {
    "Maple Cafe": "cafe", "Willow Library": "library", "Oak Hill College": "college",
    "Riverside Park": "park", "Harbor Grocery": "grocery", "Rose Pub": "pub",
    "Town Hall": "townhall", "Pine Pharmacy": "pharmacy", "Lakeside Gym": "gym",
    "Elm Art Studio": "studio", "Sunrise Bakery": "bakery", "Cedar Clinic": "clinic",
}

VISIT = {
    "cafe": ["drinking a latte by the window", "chatting with friends over coffee",
             "reading the newspaper with a cappuccino", "eating a croissant", "working on a laptop"],
    "library": ["reading a novel", "studying for an exam", "browsing the history section",
                "reading magazines", "borrowing a cookbook"],
    "college": ["attending a public lecture", "visiting the college museum"],
    "park": ["jogging along the river", "walking a dog", "feeding the ducks",
             "sitting on a bench in the sun", "having a picnic", "playing frisbee"],
    "grocery": ["buying groceries", "choosing fresh fruit", "paying at the cash register"],
    "pub": ["having a beer with friends", "watching football on television", "playing darts",
            "eating fish and chips"],
    "townhall": ["paying a parking ticket", "asking about a building permit"],
    "pharmacy": ["picking up a prescription", "buying vitamins"],
    "gym": ["lifting weights", "running on the treadmill", "doing yoga", "swimming in the pool"],
    "studio": ["taking a pottery class", "looking at paintings"],
    "bakery": ["buying fresh bread", "eating a cinnamon roll"],
    "clinic": ["waiting for a checkup", "getting a flu vaccine"],
}

HOME = {
    "morning": ["eating breakfast", "taking a shower", "making coffee in the kitchen",
                "reading the news on the phone"],
    "evening": ["cooking dinner", "watching television", "doing the dishes",
                "calling family on the phone", "reading a book on the sofa", "playing video games"],
    "night": ["sleeping"],
}

WORK = {
    "cafe_owner": ["serving coffee to customers", "baking muffins in the kitchen",
                   "taking orders at the counter", "counting the money in the register"],
    "barista": ["brewing coffee for customers", "cleaning the espresso machine",
                "making a latte", "wiping the tables"],
    "librarian": ["shelving returned books", "helping a visitor find a book",
                  "cataloguing new books", "organizing the reading corner"],
    "professor": ["teaching a lecture on medieval history", "grading essays in the office",
                  "meeting students during office hours", "writing a research article"],
    "student_chem": ["attending a chemistry lecture", "working in the chemistry lab",
                     "studying organic chemistry"],
    "student_econ": ["attending an economics lecture", "studying for a statistics exam",
                     "working on a group project"],
    "student_lit": ["attending a literature seminar", "writing an essay about poetry",
                    "reading a novel for class"],
    "student_bio": ["attending a biology lecture", "working in the biology lab",
                    "studying genetics"],
    "pharmacist": ["filling prescriptions", "advising a customer about medicine",
                   "checking the medicine stock"],
    "painter": ["painting a landscape", "mixing colors on a palette", "teaching a pottery class"],
    "grocer": ["stocking shelves with vegetables", "working at the cash register",
               "unloading a delivery truck"],
    "nurse": ["checking blood pressure of a patient", "giving a flu vaccine",
              "updating patient records"],
    "doctor": ["examining a patient", "writing a prescription", "reviewing test results"],
    "baker": ["kneading dough", "baking fresh bread", "decorating a cake", "selling croissants"],
    "bartender": ["pouring beer at the bar", "mixing cocktails", "cleaning glasses"],
    "assistant": ["preparing the council meeting", "answering calls from residents",
                  "organizing the mayor's calendar"],
    "clerk": ["filing permits", "registering a new business", "printing documents"],
    "trainer": ["leading a spin class", "coaching a client", "cleaning the gym equipment"],
    "developer": ["writing code on a laptop", "debugging a web application",
                  "joining a video call with clients"],
    "journalist": ["interviewing a resident", "writing a news article",
                   "taking photos for the newspaper"],
    "musician": ["practicing guitar", "writing a new song", "playing guitar on the small stage"],
    "gardener": ["planting tulips", "trimming the hedges", "watering the flower beds"],
    "writer": ["writing a novel", "researching old newspapers", "editing a manuscript"],
    "candidate": ["preparing a campaign speech", "talking to voters", "handing out flyers"],
}


@dataclass
class Resident:
    name: str
    role: str | None
    work: str | None                 # workplace
    home: str
    days: tuple[int, ...] = (0, 1, 2, 3, 4)   # weekday indices with work
    hours: tuple[float, float] = (9.0, 17.0)
    likes: dict[str, float] = field(default_factory=dict)   # leisure place -> weight
    sociability: float = 0.6
    interests: tuple[str, ...] = ()

    @property
    def first(self) -> str:
        return self.name.split()[0]


WEEK = (0, 1, 2, 3, 4)
SIX = (0, 1, 2, 3, 4, 5)

RESIDENTS = [
    Resident("Maria Lopez", "cafe_owner", "Maple Cafe", "the Lopez house", SIX, (7.0, 16.0),
             {"Riverside Park": 2, "Lakeside Gym": 1, "Rose Pub": 1}, 0.9, ("baking", "music")),
    Resident("John Carter", "barista", "Maple Cafe", "the Carter apartment", SIX, (7.0, 15.0),
             {"Rose Pub": 2, "Lakeside Gym": 2, "Riverside Park": 1}, 0.7, ("football", "coffee")),
    Resident("Emma Wilson", "librarian", "Willow Library", "the Wilson house", WEEK, (9.0, 17.0),
             {"Maple Cafe": 2, "Riverside Park": 2, "Elm Art Studio": 1}, 0.6, ("books", "history")),
    Resident("Oliver Brown", "professor", "Oak Hill College", "the Brown house", WEEK, (9.0, 16.0),
             {"Willow Library": 3, "Maple Cafe": 1, "Rose Pub": 1}, 0.5, ("history", "chess")),
    Resident("Sophia Martin", "student_chem", "Oak Hill College", "the Martin house", WEEK, (9.0, 15.0),
             {"Willow Library": 2, "Maple Cafe": 2, "Lakeside Gym": 1}, 0.7, ("chemistry", "dancing")),
    Resident("Lucas Martin", "pharmacist", "Pine Pharmacy", "the Martin house", WEEK, (9.0, 18.0),
             {"Riverside Park": 2, "Rose Pub": 1}, 0.5, ("gardening", "fishing")),
    Resident("Mia Clark", "painter", "Elm Art Studio", "the Clark apartment", (1, 2, 3, 4, 5), (10.0, 17.0),
             {"Maple Cafe": 2, "Riverside Park": 2, "Rose Pub": 1}, 0.7, ("art", "music")),
    Resident("Ethan Lewis", "grocer", "Harbor Grocery", "the Lewis house", SIX, (8.0, 17.0),
             {"Rose Pub": 2, "Riverside Park": 1}, 0.6, ("cooking", "football")),
    Resident("Ava Walker", "nurse", "Cedar Clinic", "the Walker apartment", WEEK, (8.0, 16.0),
             {"Lakeside Gym": 2, "Maple Cafe": 1, "Riverside Park": 1}, 0.7, ("health", "yoga")),
    Resident("Noah Hall", "doctor", "Cedar Clinic", "the Hall house", WEEK, (8.0, 16.0),
             {"Riverside Park": 1, "Lakeside Gym": 1, "Rose Pub": 1}, 0.4, ("medicine", "running")),
    Resident("Grace Young", "baker", "Sunrise Bakery", "the Hall house", SIX, (6.0, 14.0),
             {"Maple Cafe": 1, "Riverside Park": 2, "Elm Art Studio": 1}, 0.8, ("baking", "art")),
    Resident("Henry King", "bartender", "Rose Pub", "the King apartment", (1, 2, 3, 4, 5, 6), (16.0, 23.0),
             {"Lakeside Gym": 1, "Riverside Park": 1, "Harbor Grocery": 1}, 0.8, ("football", "music")),
    Resident("Chloe Scott", "assistant", "Town Hall", "the Scott apartment", WEEK, (9.0, 17.0),
             {"Maple Cafe": 2, "Lakeside Gym": 1, "Rose Pub": 1}, 0.8, ("politics", "fashion")),
    Resident("Samuel Green", "candidate", "Town Hall", "the Green house", SIX, (10.0, 16.0),
             {"Riverside Park": 2, "Maple Cafe": 2, "Willow Library": 1}, 0.9, ("politics", "history")),
    Resident("Lily Adams", "trainer", "Lakeside Gym", "the Adams apartment", SIX, (7.0, 15.0),
             {"Riverside Park": 2, "Maple Cafe": 1}, 0.7, ("fitness", "cooking")),
    Resident("Jack Baker", "student_econ", "Oak Hill College", "the Student House", WEEK, (9.0, 15.0),
             {"Rose Pub": 3, "Lakeside Gym": 1, "Maple Cafe": 1}, 0.8, ("economics", "football")),
    Resident("Zoe Nelson", "student_lit", "Oak Hill College", "the Student House", WEEK, (9.0, 15.0),
             {"Willow Library": 2, "Maple Cafe": 2, "Elm Art Studio": 1}, 0.6, ("poetry", "theatre")),
    Resident("Anna Novak", "student_bio", "Oak Hill College", "the Student House", WEEK, (9.0, 15.0),
             {"Riverside Park": 2, "Lakeside Gym": 1, "Rose Pub": 1}, 0.6, ("biology", "hiking")),
    Resident("Ryan Hill", "developer", "Maple Cafe", "the Hill apartment", WEEK, (10.0, 18.0),
             {"Lakeside Gym": 1, "Rose Pub": 1, "Willow Library": 1}, 0.4, ("programming", "games")),
    Resident("Nora Campbell", "journalist", "Town Hall", "the Campbell house", WEEK, (9.0, 17.0),
             {"Maple Cafe": 2, "Rose Pub": 1, "Riverside Park": 1}, 0.9, ("politics", "photography")),
    Resident("Leo Mitchell", "musician", "Rose Pub", "the Clark apartment", (4, 5), (20.0, 23.0),
             {"Maple Cafe": 2, "Riverside Park": 1, "Elm Art Studio": 1}, 0.7, ("music", "jazz")),
    Resident("Ruby Perez", "gardener", "Riverside Park", "the Perez house", WEEK, (7.0, 15.0),
             {"Sunrise Bakery": 1, "Maple Cafe": 1, "Harbor Grocery": 1}, 0.6, ("gardening", "nature")),
    Resident("Adam Roberts", "clerk", "Town Hall", "the Roberts house", WEEK, (9.0, 17.0),
             {"Rose Pub": 1, "Lakeside Gym": 1, "Riverside Park": 1}, 0.4, ("chess", "football")),
    Resident("Ella Turner", "writer", "Willow Library", "the Turner apartment", WEEK, (10.0, 16.0),
             {"Maple Cafe": 2, "Riverside Park": 1, "Elm Art Studio": 1}, 0.5, ("books", "theatre")),
    Resident("Max Phillips", None, None, "the Phillips house", (), (0.0, 0.0),
             {"Riverside Park": 3, "Rose Pub": 2, "Maple Cafe": 2, "Willow Library": 1}, 0.8,
             ("fishing", "history")),
]

LUNCH_PLACES = {"Maple Cafe": 2.0, "Sunrise Bakery": 1.0, "Riverside Park": 1.0}


# --------------------------------------------------------------------------- news

@dataclass
class News:
    id: str
    content: str            # what one resident tells another
    title: str              # short name used in "X told Y about ..."
    importance: int         # 1..10
    teller: str             # resident who starts spreading it
    announce: float         # hour (absolute) at which the teller learns it
    day: int | None = None  # day of the event (None: not an event)
    place: str | None = None
    start: float = 0.0      # hour of day
    end: float = 0.0
    appeal: float = 0.5     # base probability of attending when remembered
    host_spreads: bool = False   # the teller has it in their plan and always tells it
