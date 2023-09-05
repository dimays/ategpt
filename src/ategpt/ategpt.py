"""ATE-GPT: A lightweight Python library to enhance your thinking.

ATE-GPT ("Ask, Think, Evaluate" powered by GPT) is true thought 
partner, enhacing your thinking with Chain-of-Thought behavior.

This module is intended to be very lightweight, relying only on
the openai API and some simple prompt engineering techniques to 
guide a GPT-based LLM through a comprehensive thought process.

Check out www.github.com/dimays/ategpt for more information.
"""

import openai
import time
import os

# Message Templates for OpenAI ChatCompletions
SYS_MSG = """You are ATE-GPT (Ask, Think, Evaluate GPT), a powerful \
AI assistant purpose-built to help users refine their thought \
process and come to strong conclusions after a comprehensive \
chain-of-thought process.

Users will prompt you with the following information:
1. TASK: a brief description of the task they need you to complete.
2. END GOAL: a brief description of what the user hopes to finally \
achieve at the end of this thought process.
3. CONTEXT: contextual information that will help you accomplish \
the task.
4. SAMPLES: one or more samples of the kind of result they're \
looking for, each one tagged with [SAMPLE] and end-tagged with \
[ENDSAMPLE].

You MUST do the following:
1. Consider the CONTEXT carefully and reflect on it critically.
2. Complete the TASK as described, providing a final RESPONSE. \
Be sure to lean on your critical reflection of the CONTEXT, \
and match the form of any provided SAMPLES (the content between \
[SAMPLE] and [ENDSAMPLE]).
"""

USER_MSG_TEMPLATE = """TASK:
{task}

END GOAL:
{goal}

CONTEXT: 
{context}

SAMPLES:
{samples}

RESPONSE:
"""

# ------------------------
# -- Default Parameters --
# ------------------------

MODEL = 'gpt-4'

TEMP = 0.5

# -----------------------------------------------
# -- Prefixes for Each Step of the ATE Process --
# -----------------------------------------------

THOUGHT_PREFIX = "THOUGHT"

EVAL_PREFIX = "EVAL"

QUESTION_PREFIX = "QUESTION"

FINAL_THOUGHT_PREFIX = "FINAL THOUGHT"

# ----------------------------------------------
# -- Prompts for Each Task in the ATE Process --
# ----------------------------------------------

THOUGHT_TASK = """Form a new THOUGHT to continue with the ongoing \
thought process. Reflect carefully on the CONTEXT and share a new \
insight or idea that brings the thought process closer to ultimately
achieving the END GOAL. Be flexible with your thought approach and \
cater your thinking to the stated END GOAL. Be creative and bring \
in ideas from multiple discplines when applicable. If the END GOAL \
is very simple, don't overthink it--keep the response simple, too.
"""

EVAL_TASK = """Evaluate the ongoing thought process.

You MUST return EXACTLY one of the following statements:
- ASK
- THINK
- END

Here are guidelines to help you determine which statement is \
appropriate:
1. If asking yourself a QUESTION to further or more deeply explore \
the TASK, END GOAL or a recnet THOUGHT, then say "ASK", and only \
"ASK".
2. If further reflection is required, and generating another \
helpful THOUGHT would bring the thought process closer to the \
END GOAL, OR if a question was just posed and needs to be answered, \
then say "THINK", and only "THINK".
3. If you determine that the END GOAL has been met and that futher \
reflection or questioning would most likely be a waste of time, \
then say "END", and only "END".
"""

QUESTION_TASK = """Ask yourself a question about the current topic \
you are thinking about. Reflect carefully on the ultimate END GOAL \
and the overall CONTEXT, then pose a question that, if answered, \
could lead you closer to accomplishing the END GOAL.
"""

END_TASK = """Finish the current thought process by wrapping up the \
discussion with some succinct final thoughts. Be sure these final \
thoughts accurately reflect the overall CONTEXT and ensure that it \
accomplishes the END GOAL the user set out to accomplish at the \
onset of this thought process. 

Include any insights that you gain by reviewing the entire thought \
process as a whole, that you may not be aware of by just looking at \
a single thought in the process.

Do your best to keep your final thoughts and any concrete \
recommendations EASY TO UNDERSTAND. 

If the END GOAL is very simple, don't overthink it or overexplain \
it, instead keep the response simple and brief, too.
"""

# ------------------------------------------------------------
# -- Samples for ATE-GPT to Match When Completing Each Task --
# ------------------------------------------------------------

THOUGHT_SAMPLES = """[SAMPLE]
Considering our problem of sustainable urban planning, let's draw \
inspiration from nature. How can we replicate the efficient \
resource utilization seen in ecosystems to design self-sustaining \
urban environments? Think of the circulatory systems in trees that\
transport nutrients—can we apply a similar concept for resource \
distribution in cities?
[ENDSAMPLE]

[SAMPLE]
The Eiffel Tower is in Paris, France.
[ENDSAMPLE]

[SAMPLE]
In the realm of mental health, merging advancements in \
neuroscience with technology could result in personalized \
meditation apps. These apps could adapt meditation practices to an \
individual's brainwave patterns, potentially enhancing mental \
well-being more effectively.
[ENDSAMPLE]

[SAMPLE]
Jean Valjean, the central character in Victor Hugo's novel "Les \
Misérables," was arrested and imprisoned for stealing a loaf of \
bread to feed his sister's starving child.
[ENDSAMPLE]
"""

EVAL_SAMPLES = """[SAMPLE]
ASK
[ENDSAMPLE]

[SAMPLE]
THINK
[ENDSAMPLE]

[SAMPLE]
END
[ENDSAMPLE]
"""

QUESTION_SAMPLES = """[SAMPLE]
What are the key behavioral factors that influence financial \
decision-making, and how can we leverage them for better financial \
literacy?
[ENDSAMPLE]

[SAMPLE]
How can we incentivize workers to gain diverse skills and \
experiences?
[ENDSAMPLE]

[SAMPLE]
How can AI be integrated into existing art conservation \
practices without compromising the authenticity of artworks?
[ENDSAMPLE]
"""

END_SAMPLES = """[SAMPLE]
5 + 5 = 10
[ENDSAMPLE]

[SAMPLE]
Harnessing the profound insights of behavioral economics in \
financial education results in a transformative shift in how \
individuals approach their finances. Curricula designed with \
behavioral science in mind introduce concepts that resonate with \
people's natural decision-making tendencies. Gamification, \
personalized nudges, and real-world simulations make learning \
engaging and practical. The result is a populace equipped with the \
knowledge and behavioral tools to make informed financial \
decisions, achieve financial stability, and navigate complex \
economic landscapes effectively.
[ENDSAMPLE]

[SAMPLE]
The author of the Harry Potter series is J.K. Rowling.
[ENDSAMPLE]

[SAMPLE]
Adapting space exploration technologies for Earth's environmental \
challenges is an endeavor with far-reaching implications. \
Innovations such as advanced water purification systems, resilient \
materials, and sustainable energy sources have applications that \
extend beyond space. By repurposing and refining these technologies \
for terrestrial use, we not only address urgent environmental \
issues like water scarcity but also lay the foundation for a \
sustainable and technologically advanced future where humanity \
coexists harmoniously with our planet.
[ENDSAMPLE]

[SAMPLE]
The Eiffel Tower is in Paris, France.
[ENDSAMPLE]
"""


class GPTCompletion:
    def __init__(self, task, goal, prefix, context, samples,
                 model, temp, api_key=None, debug=True):
        """Utility class for creating and retrieving a Chat 
        Completion from OpenAI.
        
        Parameters:
        - task:    the particular task you need ATE-GPT to complete
        - goal:    the end-goal of the thought-process
        - prefix:  printed prefix for this task
        - context: additioanl context for ATE-GPT to consider
        - samples: samples of the kind of response you're looking for
        - model:   GPT chat model used for this completion
        - temp:    OpenAI temperature used for this completion
        - api_key: if provided, the GPTCompletion will use this value
                   for the OpenAI API key; if not provided, will look
                   for a system environment variable called 
                   OPENAI_API_KEY, default None
        - debug:   if True, prints out each step to the command line,
                   default True
        """
        self.task = task
        self.goal = goal
        self.context = context
        self.samples = samples
        self.model = model
        self.temp = temp
        self.prefix = prefix
        self.api_key = self._get_api_key(api_key)
        self.user_msg = self._construct_user_msg()
        self.completion = self._get_completion()
        if debug == True:
            print(f"{self.prefix}: {self.completion}\n\n")

    def _construct_user_msg(self):
        """Constructs the user message required to generate the
        OpenAI Chat Completion."""
        msg = USER_MSG_TEMPLATE.format(
            task=self.task,
            goal=self.goal,
            context=self.context,
            samples=self.samples
        )
        return msg
    
    def _get_completion(self, num_retries=3):
        """Retrieves the completion from OpenAI, retrying up to 3
        times if OpenAI throws an error (such as a rate limit 
        error)."""
        msgs = [
            {
                "role": "system",
                "content": SYS_MSG
            },
            {
                "role": "user",
                "content": self.user_msg
            }
        ]
        try:
            openai.api_key = self.api_key
            raw_completion = openai.ChatCompletion.create(
                model=self.model,
                temperature=self.temp,
                messages=msgs
            )
        except Exception as e:
            print(f"Completion failed: {e}\n\n")
            if num_retries > 0:
                print("Retrying in 5 seconds...")
                time.sleep(5)
                return self._get_completion(num_retries-1)
            else:
                print("No more retries.")
                raise ValueError("Unable to return completion.")
        completion = raw_completion['choices'][0]['message']['content']
        return completion

    def _get_api_key(self, api_key=None):
        """Helper function to use the provided api_key as the OpenAI
        API key (if provided), or fallback to finding the system
        environment variable OPENAI_API_KEY."""
        if api_key:
            return api_key
        return os.getenv('OPENAI_API_KEY')

class Thought(GPTCompletion):
    def __init__(self, goal, context, debug, model=MODEL, temp=TEMP):
        """GPT Completion specific to generating a unique THOUGHT
        as part of a larger thought process."""
        super().__init__(
            task=THOUGHT_TASK,
            goal=goal,
            prefix=THOUGHT_PREFIX,
            context=context,
            samples=THOUGHT_SAMPLES,
            model=model,
            temp=temp,
            debug=debug
        )

class Evaluation(GPTCompletion):
    def __init__(self, goal, context, debug, model=MODEL, temp=0):
        """GPT Completion specific to evaluating the current
        state of the thought process and determining the next 
        step."""
        super().__init__(
            task=EVAL_TASK,
            goal=goal,
            prefix=EVAL_PREFIX,
            context=context,
            samples=EVAL_SAMPLES,
            model=model,
            temp=temp,
            debug=debug
        )
        self.prefix = "EVAL"

class Question(GPTCompletion):
    def __init__(self, goal, context, debug, model=MODEL, temp=TEMP):
        """GPT Completion specific to asking a qusetion based on
        the current thought process in order to move closer to the
        end goal."""
        super().__init__(
            task=QUESTION_TASK,
            goal=goal,
            prefix=QUESTION_PREFIX,
            context=context,
            samples=QUESTION_SAMPLES,
            model=model,
            temp=temp,
            debug=debug
        )

class FinalThought(GPTCompletion):
    def __init__(self, goal, context, debug, model=MODEL, temp=TEMP):
        """GPT Completion specific to concluding the thought process
        with a single final thought."""
        super().__init__(
            task=END_TASK,
            goal=goal,
            prefix=FINAL_THOUGHT_PREFIX,
            context=context,
            samples=END_SAMPLES,
            model=model,
            temp=temp,
            debug=debug
        )

class ThoughtProcess:
    def __init__(self, goal, max_steps=10, api_key=None, debug=True,
                 model=MODEL, temp=TEMP):
        """This object represents the overall thought process. It
        will conclude once ATE-GPT determines the stated end goal
        has been reached OR once the provided number of `max_steps`
        have been taken through the thought process.
        
        Parameters:
        - goal:      the end-goal of the thought-process
        - max_steps: the maximum number of steps ATE-GPT will take
                     in this thought process (each individual task
                     is a step).
        - api_key:   if provided, each GPTCompletion will use this value
                     for the OpenAI API key; if not provided, will look
                     for a system environment variable called 
                     OPENAI_API_KEY, default None
        - debug:     if True, prints out each step to the command 
                     line, default True
        - model:     GPT chat model used for this completion, 
                     default 'gpt-4'
        - temp:      OpenAI temperature used for this completion, 
                     default 0.5"""
        self.goal = goal
        self.api_key = api_key
        self.debug = debug
        self.model = model
        self.temp = temp
        self.context = "START: The thought process has begun.\n\n"
        self.intermdiate_steps = [self.context.strip()]
        self.final_thought = ""
        self.steps_remaining = max_steps

    def think(self):
        """Generates a thought, adds it to the ongoing context
        for this thought process, then moves on to the next 
        step."""
        thought = Thought(
            goal=self.goal, 
            context=self.context,
            model=self.model, 
            temp=self.temp,
            debug=self.debug
        )
        self._add_context(thought.prefix, thought.completion)
        return self._get_next_step()

    def evaluate(self):
        """Evaluates the thought process, and recommends the next
        step, adds it to the ongoing context for this thought 
        process, then moves on to the next step.
        
        Note: the temp of this task is always 0."""
        eval = Evaluation(
            goal=self.goal,
            context=self.context,
            model=self.model,
            debug=self.debug
        )
        self._add_context(eval.prefix, eval.completion)
        return self._get_next_step()

    def ask(self):
        """Asks a question to further the thought process, adds
        it to the ongoing context for this thought process, then
        moves on to the next step."""
        question = Question(
            goal=self.goal,
            context=self.context,
            model=self.model,
            temp=self.temp,
            debug=self.debug
        )
        self._add_context(question.prefix, question.completion)
        return self._get_next_step()
        
    def end(self):
        """Ends the current thought process by generating a single
        final thought, adds it to the context for this thought
        process, then returns without moving on to the next step."""
        final = FinalThought(
            goal=self.goal,
            context=self.context,
            model=self.model,
            temp=self.temp,
            debug=self.debug
        )
        self._add_context("END", "The thought process has ended.")
        self._add_context(final.prefix, final.completion)
        return

    def _get_next_step(self):
        """Helper function for determining the next step based on
        the final word of the ongoing thought process.
        
        If thought process has reached max_steps or the latest
        evaluation returns "END", then thought process ends.
        
        If latest evaluation returns "THINK", then a new thought
        is produced.
        
        If latest evaluation returns "ASK", then a new question
        is asked.
        
        Otherwise, if latest step was not an evaluation, then
        the thought process is evaluated to determine the next
        step."""
        self.steps_remaining -= 1
        cln_context = self.context.upper().replace('"', "").strip()
        next_step = cln_context.split(" ")[-1]
        if next_step == "END" or self.steps_remaining == 0:
            return self.end()
        elif next_step == "THINK":
            return self.think()
        elif next_step == "ASK":
            return self.ask()
        else:
            return self.evaluate()
        
    def _add_context(self, prefix, completion):
        """Adds the provided prefix and completion to the context
        of the ongoing thought process as well as the list of
        intermediate steps or the final_thought, depending on how
        many steps remain.
        
        eg. THOUGHT: This is an example thought."""
        current_step = f"{prefix}: {completion}"
        self.context += f"{current_step}\n\n"
        if self.steps_remaining == 0 or prefix == FINAL_THOUGHT_PREFIX:
            self.final_thought = current_step
        else:
            self.intermdiate_steps.append(current_step)
        return
    

def main():
    """If this module is called as an executable, ATE-GPT can be 
    used in the command line with a barebones CLI that uses
    ATE-GPT's default parameters to achieve a single end goal."""
    
    print('Hello, I am ATE-GPT ("Ask, Think, Evaluate" powered by GPT), your AI thought partner! What should we think about? Please state your desired end goal below.\n')
    goal = input("MY END GOAL: ")
    print("")
    ate = ThoughtProcess(goal)
    ate.think()

if __name__ == '__main__':
    main()