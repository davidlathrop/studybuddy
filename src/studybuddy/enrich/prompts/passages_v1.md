Task: enrich passages P{first_id}-P{last_id} of the source above. Only these passages; the rest of the source is context.

Chapters of this source:
{chapters}

Return one entry per passage, in order, with:
- passage_id: the number of the passage.
- context: one sentence placing the passage in the source: the topic, the chapter, and what this passage is about. It must make sense on its own in a search result.
- summary: one plain sentence stating the passage's main point.
- keywords: 5-10 terms a student might type to find this passage: key terms, names, figures with their units, acronyms and their expansions, and everyday words for technical terms. Lowercase except proper nouns.
- questions: 8-12 distinct questions this passage answers, phrased the way a student would ask them while studying or on an exam: definitions, why and how, comparisons, applications, and plain-language paraphrases that use different words from the passage. Each has an answer_quote: the verbatim excerpt (one or two sentences) of this passage that answers it. Only include questions the passage actually answers.
- concepts: the course concepts this passage covers; skip everyday words that are only mentioned in passing. label is the standard name of the concept, surface the wording used in the passage, aliases the acronyms, expansions and alternative names. role is defines (the passage states what it is), explains (how or why it works), example (gives an instance of it) or mentions. quote is the verbatim excerpt where it appears.
- relations: relationships between concepts that this passage states explicitly, using only these types: is_a, part_of, prerequisite_of, causes, used_for, example_of, contrasts_with. head and tail are concept labels, and quote is the verbatim excerpt that states the relationship. Leave the list empty when the passage states none.

A passage with little course content (greetings, logistics) gets fewer questions and may have no concepts.
