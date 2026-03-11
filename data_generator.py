import random
import csv


def generate_classification_dataset(output_file='generated_5000_dataset.csv'):
    # --- Components for Non-Toxic Content (Label 0) ---
    nt_intros = [
        "I recently visited the newly opened restaurant downtown.",
        "The latest software update for the platform was deployed last night.",
        "I spent the weekend reading the book you lent me.",
        "Our team successfully completed the project ahead of schedule.",
        "The weather has been surprisingly pleasant this entire week.",
        "I attended the virtual workshop on machine learning yesterday.",
        "The vacation we took to the mountains was incredibly relaxing.",
        "I bought this ergonomic chair for my home office setup.",
        "The documentary about marine biology was fascinating to watch.",
        "I finally got around to fixing the leaky faucet in the kitchen.",
        "We went to the new art exhibit at the city museum.",
        "I tried out that new recipe for sourdough bread.",
        "My recent trip to the local botanical garden was delightful.",
        "I just finished assembling the new bookshelf for the living room.",
        "The annual community bake sale was a huge success this year.",
        "I downloaded that new productivity app you recommended."
    ]

    nt_bodies = [
        "The attention to detail was evident from the very beginning, making the whole experience seamless.",
        "Everything functioned exactly as advertised, which is a rare and welcome surprise these days.",
        "I was particularly impressed by how intuitive and user-friendly the entire process was.",
        "It provided a lot of valuable insights that I hadn't considered before.",
        "The quality exceeded my expectations, especially considering the reasonable price point.",
        "We encountered a few minor challenges, but they were quickly resolved without much hassle.",
        "The atmosphere was calm and inviting, perfect for unwinding after a long day.",
        "I found myself completely engrossed in it, losing track of time entirely.",
        "It required a bit of patience at first, but the end result was well worth the effort.",
        "The support staff was incredibly helpful and guided us through every step of the way.",
        "The pacing was fantastic, keeping me engaged from the first minute to the last.",
        "The aesthetic was beautiful, with vibrant colors and excellent composition.",
        "The instructions were incredibly clear and easy to follow throughout the process.",
        "Everyone involved was highly professional and courteous to all the guests.",
        "The performance was exceptionally smooth, with absolutely no lag or stuttering.",
        "It was surprisingly affordable given the high level of craftsmanship involved."
    ]

    nt_outros = [
        "I highly recommend it to anyone looking for a reliable and enjoyable experience.",
        "I'm looking forward to exploring more of what they have to offer in the future.",
        "Overall, it was a fantastic decision and I couldn't be happier with the outcome.",
        "I will definitely be sharing this with my friends and colleagues.",
        "It's a great addition to my daily routine and I plan to keep using it.",
        "Thank you so much for the recommendation, it really made my day.",
        "I consider this a major success and a step in the right direction.",
        "It was thoroughly enjoyable from start to finish.",
        "I am very pleased with the results and will be returning soon.",
        "This has completely changed my perspective for the better.",
        "I would definitely give it a five-star rating.",
        "It was a truly memorable experience that I will cherish.",
        "I can confidently say this is one of the best purchases I've made all year.",
        "It brought a lot of joy to our family and we can't wait to do it again.",
        "I will certainly be keeping a close eye on their future projects.",
        "If you're on the fence about trying it, I highly encourage you to go for it."
    ]

    # --- Components for Toxic Content (Label 1) ---
    
    # 1. Toxic Late Insults (Start normal, end aggressive)
    t_late_outros = [
        "Too bad you're too incompetent to ever achieve something like this yourself.",
        "Which just goes to show how utterly useless and pathetic your previous attempts were.",
        "Of course, explaining this to you is pointless since you have the intellectual capacity of a wet sponge.",
        "I would suggest you try it, but we both know you'd just ruin it like you ruin everything else.",
        "It's a shame your brain is too small to appreciate the nuances of something so well put together.",
        "Honestly, the fact that you can't understand this proves what an absolute moron you are.",
        "But I guess someone as lazy and worthless as you wouldn't care about quality anyway.",
        "Just seeing it made me realize how much of a disgusting failure you are by comparison.",
        "It makes your work look like it was done by an untalented, blind infant.",
        "Enjoying it almost made me forget that you are a miserable parasite dragging us all down.",
        "Then again, a braindead idiot like you wouldn't know quality if it slapped you in the face.",
        "It really highlights what a spectacular disappointment you are to everyone around you.",
        "It's fascinating how it highlights your complete inability to do anything right.",
        "Too bad your repulsive personality completely overshadows any small success you have.",
        "I'd invite you to join us next time, but nobody actually wants you around.",
        "It serves as a stark reminder of what an absolute failure you are."
    ]

    # 2. Toxic Subtle (Gaslighting, condescending, passive-aggressive)
    t_sub_intros = [
        "It's so brave of you to present a report that is this completely devoid of facts.",
        "I really admire how you don't let your profound lack of talent stop you from trying.",
        "Bless your heart for thinking this was an appropriate solution to our problem.",
        "I must say, your ability to confidently speak about things you clearly don't understand is astounding.",
        "It's truly fascinating to watch you navigate a professional environment with such glaring incompetence.",
        "I was reviewing your recent code commits and I have to applaud your commitment to writing garbage.",
        "Your latest email was a fascinating journey into the mind of someone who has given up on logic.",
        "I see you wore that outfit again today, which is a bold choice for someone in your position.",
        "I noticed you decided to speak up in the meeting today, which was certainly a choice.",
        "It takes a lot of nerve to submit a proposal that looks like a first draft.",
        "I'm always amazed by your ability to completely ignore all constructive feedback."
    ]

    t_sub_bodies = [
        "Most people would feel embarrassed presenting something so fundamentally flawed, but you just breeze right past it.",
        "The way you managed to completely miss the point of the assignment while still submitting a massive document is a unique skill.",
        "I suppose in a world with zero standards, this could be considered a moderate success.",
        "You've successfully lowered the bar so far that it's practically a tripping hazard in hell.",
        "It takes a special kind of person to drain the energy out of a room just by explaining their thought process.",
        "Your ability to turn a simple five-minute task into a week-long disaster is nothing short of miraculous.",
        "Watching you attempt to solve basic problems is like watching a dog try to understand a magic trick.",
        "The sheer volume of mistakes you make on a daily basis is honestly an impressive statistical anomaly.",
        "Your consistent dedication to delivering the bare minimum is truly something to behold.",
        "It’s almost impressive how you manage to make every single task exponentially more difficult.",
        "The fact that you thought this was acceptable work speaks volumes about your basic intelligence."
    ]

    t_sub_outros = [
        "We will definitely file this under 'cautionary tales' for the new hires.",
        "I eagerly await the day when your contributions actually match your inflated ego.",
        "Please don't strain yourself trying to fix it; I'd rather have someone capable handle it.",
        "I suppose we should be grateful that you at least managed to spell your own name correctly.",
        "Let's just pretend this never happened, for both our sakes.",
        "I'm sure one day you'll figure out how to be useful, but today is clearly not that day.",
        "Do us all a favor and try not to touch anything important for the rest of the week.",
        "Your mere presence makes everyone else's job significantly harder.",
        "I suppose we'll just have to lower our expectations even further when dealing with you.",
        "Please feel free to ask for help next time, since you clearly have no idea what you're doing.",
        "I genuinely hope one day you realize how exhausting it is for the rest of us to cover for you."
    ]

    # --- Generate Combinations ---
    # Non-Toxic pool (12 x 12 x 12 = 1,728 possible combinations)
    non_toxic_pool = [f"{i} {b} {o}" for i in nt_intros for b in nt_bodies for o in nt_outros]
    
    # Toxic Late Insult pool (12 x 12 x 12 = 1,728 possible combinations)
    toxic_late_pool = [f"{i} {b} {o}" for i in nt_intros for b in nt_bodies for o in t_late_outros]
    
    # Toxic Subtle pool (8 x 8 x 8 = 512 possible combinations)
    toxic_subtle_pool = [f"{i} {b} {o}" for i in t_sub_intros for b in t_sub_bodies for o in t_sub_outros]
    
    toxic_combined_pool = toxic_late_pool + toxic_subtle_pool

    # --- Sample exactly 2500 of each to guarantee no duplicates ---
    random.seed(42) # Ensure reproducibility 
    selected_non_toxic = random.sample(non_toxic_pool, 2500)
    selected_toxic = random.sample(toxic_combined_pool, 2500)

    # Combine into a single dataset list with labels
    dataset = [(text, 0) for text in selected_non_toxic] + [(text, 1) for text in selected_toxic]
    random.shuffle(dataset) # Shuffle so 1s and 0s are interleaved naturally

    # --- Write to CSV ---
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        # csv.QUOTE_NONNUMERIC ensures ONLY the text strings are wrapped in double quotes
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        # Write header (we don't want the header quoted, so we manually write the first line)
        f.write("text,label\n")
        
        for text, label in dataset:
            writer.writerow([text, label])
            
    print(f"Successfully generated 5000 long-form text entries and saved to '{output_file}'!")

if __name__ == "__main__":
    generate_classification_dataset()