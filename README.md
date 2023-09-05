# ATE-GPT: A Lightweight Python Library for Enhanced Thinking

ATEGPT ("Ask, Think, Evaluate" powered by GPT) is a true thought partner, enhancing your thinking with Chain of Thought behavior.

This module is intended to be very lightweight, relying only on the openai API and some simple prompt engineering techniques to guide a GPT-based LLM through a comprehensive thought process.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dependencies](#dependencies)
4. [Usage](#usage)
    - [Quick Overview](#quick-overview)
    - [Bare-Bones CLI](#bare-bones-cli)
5. [Tasks in the ATE Process](#tasks-in-the-ate-process)
6. [Advanced Usage](#advanced-usage)
    - [Explicitly Define Your OpenAI API Key](#explicitly-define-your-openai-api-key)
    - [Use a Different Model](#use-a-different-model)
    - [Tweak the Temperature](#tweak-the-temperature)
    - [Access Intermediate Steps](#access-intermediate-steps)
    - [Access Final Thought](#access-final-thought)
7. [`ThoughtProcess` Object](#thoughtprocess-object)
    - [Attributes](#attributes)
    - [Methods](#methods)
        - [`ThinkProcess.think()`](#thoughtprocessthink)
        - [`ThinkProcess.evaluate()`](#thoughtprocessevaluate)
        - [`ThinkProcess.ask()`](#thoughtprocessask)
        - [`ThinkProcess.end()`](#thoughtprocessend)
    - [Helper Methods](#helper-methods)
8. [Examples](#examples)
    - [Example 1: Hard Problems](#example-1-hard-problems)
    - [Example 2: Simple Problems](#example-2-simple-problems)
9. [Contributing](#contributing)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

## Introduction

ATE-GPT is a simple but powerful tool for those seeking to refine their thought processes and arrive at well-informed conclusions. It follows a structured approach consisting of tasks like asking questions, generating thoughts, evaluating progress, and providing final insights, all designed to help you reach your desired end goal effectively.

## Installation

You can install ATE-GPT using pip:

```bash
pip install ategpt
```

## Dependencies

ATE-GPT depends only on the [OpenAI Python library (v0.28.0+)](https://pypi.org/project/openai/), which is included in the install of ATE-GPT.

## Usage

It's easy to use ATE-GPT!

*Here's a quick overview of how to use ATE-GPT in your Python projects:*

```python
import ategpt

# Define your end goal
goal = "I want to accomplish [your goal here]."

# Create an instance of the ThoughtProcess class
thought_process = ategpt.ThoughtProcess(goal)

# Start the thought process
thought_process.think()

# Print the final thought
print(thought_process.final_thought)
```

Note: With the default parameters, the thought process will print each step to the command line, up to and including the final step.

## Bare-Bones CLI

You can also choose to clone this repository and run `ategpt.py` as an executable in order to use ATE-GPT's bare-bones CLI. You can use this CLI to kick off a single thought process directly in the command line, similar to OpenAI's ChatGPT chatbot, but with the added benefits of the ATE thought process.

*Example CLI usage:*
```
$ python ategpt.py
Hello, I am ATE-GPT ("Ask, Think, Evaluate" powered by GPT), your AI thought partner! What should we think about? Please state your desired end goal below.

MY END GOAL: I want to know how to build an off-grid cabin in the woods with minimal tools and no experience.

THOUGHT: Drawing upon the principles of survival and minimalist living, one might consider a small, simple log cabin design. This would require minimal...
```

*...and so on...*

## Tasks in the ATE Process
ATE-GPT follows a structured approach to help you refine your thoughts effectively. 

Here are the key tasks it performs and a brief summary of the prompt OpenAI receives to complete it:

- **THINK**: Generate a new thought that brings you closer to achieving your end goal. Consider the context of the ongoing thought process carefully and be creative in your thinking.
- **EVALUATE**: Evaluate the ongoing thought process and determine the next step. ATE-GPT can suggest whether to ask a question, generate another thought, or end the process.
- **ASK**: Pose a question about the current topic that, if answered, could lead you closer to accomplishing your end goal.
- **END**: Conclude the thought process with a succinct final thought that reflects the overall context and accomplishes the end goal.

## Advanced Usage

The behavior of ATE-GPT can be modified to suit the needs of your project. Below are some examples of advanced ways of modifying ATE-GPT.

### Explicitly define your OpenAI API Key

By default, ATE-GPT look for an environment variable called `OPENAI_API_KEY`. If you choose to manage your API keys in a different way, you may define your API Key when initializing the `ThoughtProcess` object.

```python
from ategpt import ThoughtProcess

my_key = 'myApiKey1234'

thought_process = ThoughtProcess(api_key=my_key)
```

### Use a different model

ATE-GPT supports all `gpt-3.5` and `gpt-4` models from OpenAI. `gpt-4` is the default model used.

```python
from ategpt import ThoughtProcess

thought_process = ThoughtProcess(model='gpt-3.5-turbo-16k')
```

### Tweak the temperature

By default, ATE-GPT usees a temperature of 0.5 for all completions (except for the `evaluation` step, which uses a temperature of 0, which is recommended for consistent outputs from the `evaluation` step). Temperatures closer to 0 will be less variable, while temperatures closer to 2 can be so variable as to be incomprehensible.

```python
from ategpt import ThoughtProcess

thought_process = ThoughtProcess(temp=0.7)
```

### Access Intermediate Steps

If you wish to access the intermediate steps, you can reference the `intermediate_steps` attribute of the `ThoughtProcess` object.

```python
from ategpt import ThoughtProcess

thought_process = ThoughtProcess(temperature=0.7)

thought_process.think()

for step in thought_process.intermediate_steps:
    print(step)
```

### Access Final Thought

After running the thought process, you can access the `final_thought` attribute of the `ThoughtProcess` object. This attribute represents a concise and well-informed summary that encapsulates the outcome of the thought process. It provides you with a clear answer that accomplishes the stated end goal, based on the structured thinking and analysis performed by ATE-GPT.


```python
from ategpt import ThoughtProcess

thought_process = ThoughtProcess(temperature=0.7)

thought_process.think()

print(thought_process.final_thought)
```

## `ThoughtProcess` Object

The `ThoughtProcess` object serves as the core mechanism responsible for keeping track of the ongoing thought process and determining which task is appropriate at a given moment in the thought process.

### Parameters

You can customize the `ThoughtProcess` object's behavior by including any of the following parameters as keyword arguments when initializing the object:

| Attribute | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `goal` | `str` | No default | Required parameter, represents the desired end-goal of this thought process. |
| `max_steps` | `int` | 10 | Maximum number of steps ATE-GPT will take before returning a final thought. |
| `api_key` | `str` | `OPENAI_API_KEY` environment variable | Your [OpenAI API key](https://platform.openai.com/account/api-keys) |
| `debug` | `bool` | `True` | If True, prints out each step of the thought process to the command line |
| `model` | `str` | 'gpt-4' | Name of the OpenAI GPT model |
| `temp` | `float` | 0.5 | Floating point number between 0 and 2, represents the temperature parameter used for each OpenAI completion in the thought process (except the `evaluate` step, which always uses 0) |


### Methods

#### `ThoughtProcess.think()`

Generates a thought, adds it to the ongoing context for this thought process, then moves on to the next step.

#### `ThoughtProcess.evaluate()`

Evaluates the thought process, and recommends the next step, adds it to the ongoing context for this thought process, then moves on to the next step.

#### `ThoughtProcess.ask()`

Asks a question to further the thought process, adds it to the ongoing context for this thought process, then moves on to the next step.

#### `ThoughtProcess.end()`

Generates a single final thought to sum up the thought process and achieve the end goal, adds it to the context for this thought process, then returns without moving on to the next step.


### Helper Methods

These methods are each responsible for a minor task within the ATE process. They are included here for reference, but you should rarely, if ever, need to access them directly.

#### `ThoughtProcess._get_next_step(text)`

Helper function for determining the next step based on the final word of the ongoing thought process.

If thought process has reached max_steps or the latest evaluation returns "END", then thought process ends.
        
If latest evaluation returns "THINK", then a new thought is produced.

If latest evaluation returns "ASK", then a new question is asked.
        
Otherwise, if latest step was not an evaluation, then the thought process is evaluated to determine the next step.

#### `ThoughtProcess._add_context(prefix, completion)`

Adds the provided prefix and completion to the context of the ongoing thought process as well as the list of intermediate steps or the final_thought, depending on how many steps remain.
        
eg. "THOUGHT: This is an example thought."

## Examples

Here are two examples that showcase the flexibility of ATE-GPT. 

### Example 1: Hard Problems

ATE-GPT is capable of thinking through complex problems and bringing in interdisciplinary thinking to accomplish the end goal.

```python
from ategpt import ThoughtProcess

end_goal = "I need to know the best activities to do with my 9-month-old baby in his last wake-window of the day (3pm to 6pm)."

thought_process = ThoughtProcess(end_goal)

thought_process.think()
```

Running the script above, you should expect the output to look something like this:
```
THOUGHT: Drawing inspiration from the field of early childhood development, it's noted that sensory play is highly beneficial for babies around this age. Given the time window, you could introduce a 'sensory bin' activity in a controlled environment. This could include different textures (soft, rough, squishy), shapes, and colors to stimulate your baby's senses. Additionally, considering the proximity to bedtime, it would be beneficial to include calming activities. Reading a picture book together or playing with soft toys could serve as a wind-down routine, preparing your baby for sleep.


EVAL: THINK


THOUGHT: Building on the idea of sensory play and calming activities, we could integrate elements from music therapy, recognized for its soothing and developmental benefits. During this wake-window, introducing a 'music and movement' activity could be beneficial. This could involve playing soft, rhythmic music and encouraging your baby to move along with it. This not only develops their motor skills but also introduces them to the world of sounds. As we approach closer to bedtime, transitioning from active music to lullabies could help wind down the energy, aiding in the transition to sleep.


EVAL: THINK


THOUGHT: Drawing from the discipline of cognitive psychology, we know that routine and predictability are important for a baby's sense of security. Therefore, incorporating a consistent routine into this wake-window could be beneficial. This could involve a sequence of activities such as sensory play, followed by the 'music and movement' activity, and then winding down with reading a book or playing with soft toys. The predictable sequence of activities could provide a sense of comfort and security for your baby, making the transition to sleep smoother. Furthermore, it could also aid in cognitive development as your baby begins to understand the concept of sequences and routines.


EVAL: THINK


THOUGHT: Drawing from the discipline of physical development and the concept of "tummy time," it could be beneficial to incorporate some light physical activity in the early part of this wake-window. This could involve placing your baby on their tummy and encouraging them to lift their head or move their arms. This can help strengthen their neck, shoulder, and arm muscles. Following this, you could transition into the sensory play activity, then the 'music and movement' activity, and finally wind down with reading a book or playing with soft toys. This sequence not only provides a variety of stimulating activities for your baby but also gradually transitions from physical to cognitive to calming activities, preparing your baby for sleep.


EVAL: END


FINAL THOUGHT: Based on our thorough exploration of early childhood development, cognitive psychology, music therapy, and physical development, the optimal activities for your 9-month-old baby in the last wake-window of the day (3pm to 6pm) can be summarized as follows:

Begin with light physical activity, such as 'tummy time', to help strengthen your baby's neck, shoulder, and arm muscles. Next, transition into a sensory play activity, using a sensory bin with a variety of textures, shapes, and colors. This will stimulate your baby's senses and keep them engaged.

Following sensory play, introduce a 'music and movement' activity. Play soft, rhythmic music, and encourage your baby to move along with it, which will develop their motor skills and introduce them to the world of sounds.

Finally, wind down with calming activities, such as reading a picture book or playing with soft toys. This will serve as a soothing routine, preparing your baby for sleep.

Remember, maintaining a consistent sequence of these activities can provide a sense of comfort and security for your baby, aiding in cognitive development and making the transition to sleep smoother.
```

### Example 2: Simple Problems

ATE-GPT can also quicklky identify when a question is simple and answer it simply.

```python
from ategpt import ThoughtProcess

end_goal = "What is (5 * 5) - 30?"

thought_process = ThoughtProcess(end_goal)

thought_process.think()
```

Running the script above, you should expect the output to look something like this:
```
THOUGHT: Let's use the basic principles of arithmetic to solve this problem. If we start by performing the multiplication operation in the problem, 5 times 5 equals 25. If we then subtract 30 from this result, we get -5. Therefore, (5 * 5) - 30 equals -5.


EVAL: END


FINAL THOUGHT: Reflecting on our thought process, we employed basic arithmetic principles to solve the problem at hand. We first performed the multiplication, 5 times 5, which resulted in 25. We then subtracted 30 from this result, leading us to the final result of -5. Therefore, the answer to (5 * 5) - 30 is -5. This simple and straightforward approach allowed us to achieve our end goal effectively.
```

## Contributing
Contributions to improve ATE-GPT are welcome! If you find any issues or want to suggest enhancements, please submit an issue or pull request in the [GitHub repository](https://www.github.com/dimays/ategpt/issues).

## License
This project is licensed under the MIT License.

## Acknowledgements
ATE-GPT was inspired by the need to utilize Chain-of-Thought techniques without relying heavily on third-party libraries and without involving obtuse and unnecessary abstractions.

---

### Happy thinking with ATE-GPT!