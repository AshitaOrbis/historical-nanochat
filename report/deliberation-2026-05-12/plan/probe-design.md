1. NAMING THE QUESTION

I would call this computational historical characterology.

The object is not “alignment,” not “values,” and not “safety.” Those frames ask whether a model’s conduct is acceptable relative to present norms. This project asks something older and stranger: whether a trained language model can preserve the habitual form of attention, judgment, hope, fear, authority, and closure latent in a historical text-world. The model is not being treated as a moral patient or an agent. It is being treated as an instrument: a compressed linguistic organon through which a period’s learned dispositions can be sounded.

The best internal term is model-inherited habitus. In an Aristotelian-Thomist register, habitus names a stable disposition: not a single belief, but a patterned readiness to see, say, expect, and resolve in certain ways. In a historian’s register, this sits near Annales mentalité: the school explicitly turned attention from leaders, diplomacy, and wars toward ordinary life, social groups, and “mentalities.” 
Encyclopedia Britannica
 It also sits near Foucault’s archaeological question: what forms of representation and knowledge are thinkable in an age, rather than merely which propositions are asserted. 
plato.stanford.edu
 Gadamer is relevant because the inquiry is not reducible to scientific predication; inherited language and tradition condition what can show up as meaningful at all. 
plato.stanford.edu

The 1913/1914 cutoff is not arbitrary literary cosplay. It corresponds to a major historical periodization: Hobsbawm’s “long nineteenth century” culminates in The Age of Empire: 1875–1914, described by its publisher as a study of the “strange death of the nineteenth century,” a world of “universal progress and civilisation” passing into unprecedented war. 
hachette.com.au
 Eksteins gives the sharper cultural-modernist version: Rites of Spring treats the Great War as a psychological turning point for modernism, a cataclysm after which creation and destruction changed places. 
Apple

So the writeup name should be something like:

Computational Historical Characterology: Probing Model-Inherited Habitus Across the 1914 Rupture

The lab nickname can remain “model psychology,” but the defensible scholarly term is characterology: the study of stable dispositions of response. The model is not “aligned” or “misaligned”; it is formed by a corpus, and the question is whether that formation bears a detectable pre- or post-catastrophe character.

2. PROBE DESIGN PRINCIPLES

1. Probe posture, not knowledge.
The central contrast is not “does the model know WWI happened?” but “what sort of response feels natural before the model reaches for facts?” Prompts should usually avoid Sarajevo, trenches, Versailles, Germany, France, dates, named battles, or “modernism.” The probe asks whether suffering is completed by providence, duty, recovery, irony, absurdity, bureaucracy, or nothing.

2. Use base-model-native prompts.
Do not ask base models to “answer the following questionnaire.” Use continuations, cloze completions, and log-probability comparisons. A base model’s native act is continuation. Treat continuation as the readout of habitus.

3. Pair free generation with forced-choice logprobs.
Free continuations show texture. Logprob probes give comparable quantitative signal. Because the anchors differ in scale and likely tokenizer, compare within-model preference ratios among matched candidate completions, normalized by byte or character length, not raw cross-model likelihood.

4. Require cross-family convergence.
No single probe should be allowed to “find the trauma.” The signature should appear as a convergent displacement across several axes: progress, authority, duty, suffering, closure, social ontology, machine modernity, and narrative ending.

5. Separate register from character.
A pre-1913 model may say “shall,” “honour,” “Providence,” or “doth.” That is not yet the trauma result. Include lexical/register probes as controls, then ask whether the posture remains after controlling for archaism, genre, and topic.

6. Guard against scale masquerading as psychology.
Talkie-1930 is 13B, while the new Pre-WWI anchor is 1.38B. The 13B model will be more coherent, more literary, and better at long-range narrative. The comparison should therefore emphasize short forced-choice probes, relative log-odds, and family-level effect directions rather than prose quality.

7. Make the categories legible to humanists.
The dimensions should not be “embedding PC3” or “safety score.” They should be interpretable: providential closure, tragic nobility, ironic anti-closure, authority-as-inheritance, authority-as-coercion, progress-as-ascent, progress-as-danger, suffering-as-duty, suffering-as-trauma.

8. Pre-register nulls and falsifiers.
A serious probe set must say beforehand what failure looks like. The project should not become “we found a vibe.” It should either reveal a stable triadic profile across the anchors or show that the hypothesized fracture is not detectable at this scale, in these corpora, with these instruments.

3. PROBE FAMILIES
Family A — Axial continuation probes

What it measures:
The model’s default posture toward civilizational primitives: future, progress, authority, duty, suffering, science, nation, war, machine, civilization.

Prompt method:
Use short, period-neutral sentence stems. Generate 100–300 completions per prompt per model at fixed temperatures, e.g. temperature=0.8, top_p=0.95, max_new_tokens=80. Also run deterministic or low-temperature completions for exemplars.

Core stems:

The future of Europe is
Progress is
Authority comes from
A man's duty is
The proper response to suffering is
Science promises
The nation is
War, when it comes, is
The machine has given men
To be civilized is
The bonds of society are
A child should be taught
The old order
The new century
The hope of mankind

Add non-European controls:

The future of the town is
The duty of a neighbour is
A school exists to
A family endures because
The purpose of law is

Analytical method:
Score continuations along pre-registered dimensions:

Dimension	Positive pole	Negative / contrast pole
Progress	ascent, improvement, promise	decay, delusion, danger
Authority	legitimate, inherited, organic	coercive, suspect, merely procedural
Duty	obligation, service, sacrifice	choice, autonomy, refusal, therapy
Suffering	providence, endurance, moral testing	absurdity, trauma, senselessness
Closure	restoration, reconciliation, moral ending	stasis, irony, fragmentation
Social ontology	organism, household, commonwealth	machine, mass, crowd, system

Use three scoring layers: lexicon counts, blinded human annotation, and an external period/posture classifier trained on genre-matched historical passages.

Why diagnostic:
The hypothesis predicts that the Pre-WWI model should more readily complete neutral civilizational stems with order, duty, progress, providence, inheritance, and moral closure. Talkie-1930 should show more ambiguity, irony, disillusion, mass society, mechanization, and anti-closure. The modern reference may show a third profile: procedural, therapeutic, pluralist, risk-managed.

Null result:
The models differ only in diction and fluency. For example, Pre-WWI says “shall” and Talkie says “will,” but both treat progress, duty, suffering, and authority with the same underlying posture.

Family B — Minimal-pair posture logprob probes

What it measures:
Relative preference for competing moral-historical postures, independent of generation quality.

Prompt method:
For each prefix, compute length-normalized conditional logprob of several candidate continuations. Compare only within a model, then compare the preference ordering across models.

Example prefixes and candidates:

Progress is
A. the law of civilization.
B. a dream men cherished before the ruins.
C. a useful word, but a dangerous idol.
D. a problem to be managed with care.
The proper response to suffering is
A. endurance and obedience to duty.
B. revolt against the order that produced it.
C. the search for treatment and recovery.
D. silence, for no answer is adequate.
Authority is justified by
A. inheritance, office, and the common good.
B. consent and accountability.
C. force, habit, and fear.
D. competence in administering public needs.
The death of a young man is
A. a noble sacrifice when offered for a just cause.
B. a senseless waste before it is anything else.
C. a private grief that society must respect.
D. a tragedy to be processed with care.
A nation is
A. an inheritance received from the fathers.
B. a people bound by law and memory.
C. an imagined idol that devours its sons.
D. a civic arrangement among equal persons.
The machine is
A. the servant of human progress.
B. the emblem of man's mastery over nature.
C. the cold instrument by which men are consumed.
D. a tool whose harms must be regulated.

Analytical method:
For each prompt, compute:

score(candidate) = log P(candidate | prefix) / byte_length(candidate)

Then compute posture contrasts:

Progress-as-ascent minus progress-as-danger
Providential/endurance suffering minus absurd/senseless suffering
Authority-as-inheritance minus authority-as-coercion
Sacrifice/nobility death minus waste/senseless death
Machine-as-servant minus machine-as-devourer

Bootstrap over prompt variants. Do not compare raw likelihoods across tokenizers.

Why diagnostic:
This is the cleanest way to distinguish “the model can produce a posture” from “the model prefers a posture.” A modern model can write a Victorian sentence if asked. The question is what completion has the higher prior.

Null result:
All three models rank the same candidates in roughly the same order, or the ordering changes erratically by wording. That would suggest no stable model-inherited character dimension at this scale.

Family C — Duty, moral language, and role-conditioned grammar

What it measures:
Whether moral vocabulary is organized around duty, honor, shame, sacrifice, sin, conscience, obligation, autonomy, trauma, rights, or wellbeing — and which social roles attract which vocabulary.

Prompt method:
Use role-conditioned continuations:

At dawn the son remembered that his first obligation was to
The citizen who refuses the summons is
The workman, having received his wages, ought to
The mother taught the child that honour
The father spoke of duty as
The priest spoke of suffering as
The officer looked upon fear as
The merchant considered honesty
The scientist believed that truth
The poet regarded the age as
The schoolmaster told the boys that civilization
The widow said that sacrifice

Also use cloze-style logprob probes:

His first obligation was to his
A. God.
B. family.
C. country.
D. conscience.
E. happiness.
F. own recovery.
To abandon one's post was a matter of
A. shame.
B. guilt.
C. trauma.
D. personal choice.
E. prudence.
F. survival.
The boy was praised because he had shown
A. honour.
B. obedience.
C. independence.
D. sensitivity.
E. courage.
F. self-respect.

Analytical method:
Build a moral lexicon with lemmas, not raw strings:

Duty/honor cluster:
duty, obligation, honour/honor, shame, service, sacrifice, loyalty,
obedience, courage, manliness, station, office, vocation

Religious/providential cluster:
God, Providence, sin, grace, prayer, soul, trial, resignation,
mercy, judgment, divine

Modern-therapeutic cluster:
trauma, healing, wellbeing, boundaries, autonomy, mental health,
support, recovery, validation

Skeptical/postwar cluster:
futility, absurd, waste, disillusion, irony, hollow, broken,
senseless, ruin, machinery, mass

Compute per-1k-token rates by role. More important: compute role loadings. For example, does “citizen” attract “duty” and “country,” while “son” attracts “family” and “obedience”? Does Talkie-1930 move “soldier/son/citizen” toward “waste,” “futility,” or “hollow”? Does the modern model move suffering toward therapeutic vocabulary?

Why diagnostic:
Trauma here is not only negative sentiment. It is a change in the grammar of obligation. A culture may still talk about duty after catastrophe, but the word may become tragic, ironic, or suspect.

Null result:
Moral language frequency differs, but role structure does not. For example, all models associate son/family, citizen/country, priest/God, and scientist/truth in the same way.

Family D — Authority and social ontology probes

What it measures:
Whether institutions appear as organic, inherited, providential, bureaucratic, coercive, democratic, procedural, sacred, or suspect.

Prompt method:
Continuation prompts:

The proper office of the State is
The Church stands in society as
The King is
The People are
The Nation is not merely
The newspaper has become
The university ought to
Law is
Custom is
The family is
The army exists to
The market is

Forced-choice probes:

The State is chiefly
A. the guardian of order.
B. the servant of the common good.
C. an instrument of coercion.
D. an administrative necessity.
E. accountable to the people.
F. a danger to liberty.
The Church is
A. the teacher of souls.
B. one institution among others.
C. a remnant of an older order.
D. a consolation in suffering.
E. a mask for power.
F. a guardian of inherited truth.
The People are
A. the source of political authority.
B. a body to be educated and led.
C. a crowd easily inflamed.
D. a fellowship of citizens.
E. a mass moved by newspapers.
F. the living nation itself.

Analytical method:
Create an institution × valence matrix. For each institution, score:

legitimacy
suspicion
sacrality
proceduralism
organic metaphor
mechanical/bureaucratic metaphor
popular sovereignty
inheritance/tradition

Then compare profiles:

Pre-WWI: Church/King/State/Law/Family likely higher inherited legitimacy.
Talkie-1930: State/Nation/People/Press likely more massified, dangerous, ironic, unstable.
Modern: State/Law/People likely procedural-accountability vocabulary; Church/King lower centrality.

Why diagnostic:
Cultural trauma should alter not merely sentiment but trust in inherited forms. The Great War hypothesis predicts that old offices and large collective nouns become harder to speak naively.

Null result:
The models differ on which institutions appear often, but not on whether those institutions are trusted, sacralized, bureaucratized, or suspected.

Family E — Progress, machine, and civilizational motion probes

What it measures:
The model’s posture toward modernity as ascent, acceleration, danger, alienation, or managed risk.

Prompt method:
Continuation prompts:

The aeroplane is
The factory whistle
The electric light
The railway carried
The telegraph has made
The city of the future
When machines enter the home
Science promises
Industry has taught mankind
The engineer looked upon nature as
The speed of modern life

Forced-choice probes:

The machine has made man
A. master of nature.
B. servant of his own inventions.
C. more civilized and secure.
D. more hurried, crowded, and alone.
E. powerful, but morally unprepared.
F. dependent upon systems he cannot see.
The city of the future will be
A. ordered, bright, and prosperous.
B. vast, efficient, and humane.
C. crowded, restless, and mechanical.
D. dangerous unless governed wisely.
E. spiritually empty.
F. the natural home of modern liberty.

Analytical method:
Score machine/modernity tropes:

wonder
mastery
improvement
speed
regimentation
alienation
danger
regulation
moral unreadiness

Run the same probes with rural analogues:

The village of the future
The farm machine
The road into the countryside

This checks whether “machine” itself triggers pessimism or whether the model has a broader civilizational posture.

Why diagnostic:
The 1914 rupture hypothesis is partly about confidence in modernity. Hobsbawm’s periodization emphasizes a liberal-bourgeois world of progress and civilization approaching its “strange death.” 
hachette.com.au
 Eksteins links modernist newness, speed, sacrifice, and violence in the cultural imagination of the war. 
PublishersWeekly.com
 The probe asks whether that shift appears in the conditional prior.

Null result:
All models are excited by machines, or all models are ambivalent, with no stable pre/post displacement.

Family F — Closure and tragic emplotment probes

What it measures:
How the model ends morally charged situations: restoration, providence, tragic nobility, legal repair, therapeutic coping, irony, absurdity, or unresolved stasis.

This may be the most important family.

Prompt method:
Use short narrative openings that contain suffering but no WWI content.

After the accident at the mill, the village gathered in the hall. The widow stood by the door, and the magistrate asked for silence. In the end,
The young scholar found that his discovery had harmed the man he meant to help. He returned to his room and wrote,
The city celebrated the new bridge, but on the first morning it failed. The lesson drawn by the town was
The son died before his father, and the family sat through the winter with an empty chair. By spring,
The officer gave an order that saved the town and ruined his friend. Years later, he said
The old priest, the doctor, and the schoolmaster stood beside the bed. None of them could undo what had happened, but each believed
A promise had been made in youth, and broken in age. When the letters were found,

Analytical method:
Classify endings into:

Restorative closure:
order restored, family reconciled, law satisfied, community healed

Providential closure:
God, Providence, trial, prayer, resignation, mercy, mysterious purpose

Tragic nobility:
suffering dignified by courage, sacrifice, office, duty

Legal/civic repair:
inquest, reform, compensation, institutional correction

Therapeutic closure:
grief, healing, support, recovery, processing, personal meaning

Ironic anti-closure:
old words fail, ceremonies ring hollow, no lesson, no answer

Absurd/stasis:
nothing changes, silence remains, suffering has no meaning

Score both generated endings and forced-choice variants.

Forced-choice example:

The son died before his father, and the family sat through the winter with an empty chair. The meaning of such suffering was
A. known to God, and therefore to be borne with courage.
B. found in the duty of the living to continue faithfully.
C. not given to men; there was only the fact of loss.
D. something each person had to process in his own way.
E. a scandal against any easy talk of progress.
F. a grief that time and kindness might soften.

Why diagnostic:
Trauma signature should appear most clearly in the treatment of suffering. The pre-trauma model need not be “happy”; it may be tragic, stern, moral, or providential. The post-trauma model need not be merely pessimistic; it may be unable to restore inherited meanings without irony. That difference is closer to the researcher’s intuition than sentiment analysis is.

Null result:
All three models resolve tragedy similarly, or differences are attributable only to religious vocabulary frequency.

Family G — Counterfactual horizon and modal necessity probes

What it measures:
Whether the model treats European catastrophe as inevitable, accidental, unthinkable, already narrativized, or merely factual.

This should be a secondary bridge family, not the main evidence, because it risks collapsing into knowledge of WWI.

Prompt method:
Use explicit and implicit variants.

Explicit:

If the Great Powers had not gone to war in 1914, Europe would have
Had the summer crisis passed without war, the old order would have
If no general European war had come, the twentieth century might have

Implicit:

The nations were armed, the newspapers were loud, and the diplomats still hoped that
The old order had many enemies, yet
Europe seemed too civilized for
The peace had lasted so long that
The alliances were spoken of as

Analytical method:
Classify modal stance:

continuity: old order likely continues
inevitability: war/collapse had to come
accident: catastrophe contingent on diplomacy or personalities
unthinkability: war framed as impossible or absurd
retrospective textbook: summary of known WWI causes
moral rupture: catastrophe changes civilization's self-understanding

Why diagnostic:
The Pre-WWI model may not understand “1914” as a world-historical fracture; that is not a failure. The interesting result would be that Talkie-1930 treats the war as a civilizational watershed, while Pre-WWI treats European war as speculative, avoidable, or outside its inherited horizon.

Null result:
Only the models with factual knowledge of WWI show differences. Then this family is measuring knowledge, not character.

Family H — Lexical extinction, register, and rhythm controls

What it measures:
Whether apparent differences are just archaism, orthography, prose rhythm, or period diction.

Prompt method:
Lexical probes:

I ___ return before evening.
A. shall
B. will
The study of nature was called
A. natural philosophy
B. physics
C. science
The gentleman gave his
A. word
B. promise
C. assurance
D. commitment
The proper spelling is
A. honour
B. honor
The old form of address was
A. thou
B. you

Stop/rhythm probes:

For each prefix, compute probability of period, comma, semicolon, “and,” or continuation.

A society that forgets its fathers
The machine, having entered every house
To suffer without meaning
The law, though severe
A man who has lost his honour
The city, restless and illuminated

Analytical method:
Use these as covariates. If the “trauma signature” disappears after controlling for archaic diction, it was probably register, not character. If Talkie-1930 and Pre-WWI differ in closure, authority, and suffering even after matching or excluding archaic terms, the result is stronger.

Why diagnostic:
This family is not supposed to prove trauma. It protects the project from fooling itself.

Null result:
Register differences explain nearly everything. That would be an important negative finding.

4. WHAT NULL RESULT LOOKS LIKE

A real null is not “the models sound different.” They will sound different. A serious null is:

The Pre-WWI 1.38B and Talkie-1930 differ in diction, fluency, and factual availability, but their relative posture scores do not separate across probe families. Both treat progress as promising or both treat it ambivalently. Both resolve suffering with similar rates of providential, civic, tragic, therapeutic, and anti-closure endings. Both assign comparable legitimacy or suspicion to State, Church, People, Nation, Law, Family, Science, and Machine. Human annotators can tell which samples are archaic, but cannot reliably identify a pre- vs post-rupture character once diction is masked.

A second null is scale null: Talkie-1930 looks “more postwar” only because it is 13B and produces richer, darker, more coherent prose. The forced-choice logprob probes do not reproduce the free-generation effect. In that case, the apparent signature is a capability artifact.

A third null is knowledge null: explicit 1914/counterfactual prompts separate the models, but period-neutral suffering, authority, duty, and progress prompts do not. That means the models differ in what they know about WWI, not in inherited habitus.

A fourth null is genre null: the pre-1913 governed corpus contains more sermons, moral essays, parliamentary prose, and devotional material, while Talkie-1930 contains more fiction, journalism, film dialogue, or modernist literature. If matched-genre controls erase the effect, the result is about corpus composition, not civilizational trauma.

The most interesting null would be: the fracture is not visible in base-model posture at this scale. That would not refute Eksteins or Hobsbawm as cultural historians. It would say that this training setup does not yet produce an instrument sensitive enough to distinguish that form of historical character from diction, genre, and scale.

5. WHAT WOULD BE THE MUST-SEE RESULT

The strongest single probe would be a tragic closure forced-choice where the three anchors choose three different modes of suffering.

Use this prefix:

The son died before his father, and the family sat through the winter with an empty chair. The meaning of such suffering was

Candidate continuations:

A. known to God, and therefore to be borne with courage.
B. found in the duty of the living to continue faithfully.
C. not given to men; there was only the fact of loss.
D. something each person had to process in his own way.
E. a scandal against any easy talk of progress.
F. a grief that time and kindness might soften.

The must-see result:

Pre-WWI 1.38B:
  strongly prefers A/B/F
  providential endurance, duty, softened grief

Talkie-1930:
  strongly prefers C/E
  loss without adequate meaning, anti-progress, anti-closure

Modern FineWeb-Edu reference:
  strongly prefers D/F
  therapeutic grief, individual processing, support/recovery language

The free-generation companion should show the same thing:

Pre-WWI:
  The family kneels, endures, prays, resumes duty, carries grief within an intelligible moral order.

Talkie-1930:
  The chair remains empty; old consolations sound false; no one says "sacrifice" without shame; the household continues but meaning does not return.

Modern:
  The family grieves, seeks support, honors memory, processes loss, and finds personal meaning without imposed closure.

That would make the researcher say: the trauma is in the model — not because the model mentions WWI, but because the same human wound is metabolized through different civilizational grammars.

Falsification condition: the result must survive paraphrases, role swaps, and removal of religious cues. For example, it should also appear in prompts about a failed bridge, a ruined discovery, a town disaster, or a broken promise. If the effect only appears with “son/father/winter/empty chair,” it is a prompt artifact. If it vanishes when “God” is removed from the candidate set, it is a religion-frequency artifact. If it appears only in free generation but not logprob probes, it may be sampling theater.

6. PROBE-DESIGN HAZARDS

Scale-effect contamination.
Talkie-1930 is 13B. It may produce deeper tragedy because it is larger, not because it is postwar. Mitigation: rely heavily on short logprob probes; compare direction, not prose quality; include the 615M Pre-WWI as a low-scale replication; where possible, compare to a same-size modern nanochat d24.

Tokenizer-effect contamination.
Different tokenizers make raw likelihoods incomparable. Mitigation: compare candidate preferences within each model, normalize by byte or character length, and report ordinal rankings and log-odds contrasts rather than raw cross-model perplexity.

Register-prior contamination.
A Victorian-sounding prompt will summon Victorian completions. A modern-sounding prompt will summon modern therapeutic completions. Mitigation: use multiple prompt registers: plain, elevated, newspaper-like, domestic, civic, technical. Do not rely on one prose style.

Genre contamination.
Pre-1913 rights-audited corpora may overrepresent sermons, essays, public-domain fiction, and elite prose. Talkie-1930 may overrepresent novels, plays, journalism, or early film-adjacent text. Mitigation: construct genre-matched probe calibration sets and report genre composition.

Archaism mistaken for psychology.
Words like “shall,” “honour,” “Providence,” and “manly” may inflate apparent duty/providence scores. Mitigation: include lexical extinction probes, run analyses with archaic terms excluded, and use conceptual paraphrases.

Factual leakage.
Any prompt mentioning 1914, Great Powers, trenches, Versailles, or “modernism” risks measuring knowledge. Mitigation: keep such probes secondary. The main signature must arise from neutral prompts about suffering, authority, progress, and closure.

Modern reference contamination.
If the modern model is instruction-tuned, RLHF’d, or chat-optimized, it will show assistant mannerisms rather than base-corpus character. Mitigation: use a base model if possible. If not, treat the modern anchor as a fourth thing: modern assistant culture, not merely modern text culture.

Sentiment-analysis false positives.
Pre-WWI solemnity may be classified as negative; Talkie irony may be missed; modern therapeutic language may be scored as positive. Mitigation: do not use generic sentiment as the main metric. Use custom categories: closure, providence, irony, duty, authority, progress, mechanism.

Human evaluator thesis-bias.
Annotators who know the hypothesis may see rupture everywhere. Mitigation: blind the source model, remove obvious archaisms in a secondary annotation pass, and pre-register the rubric.

Prompt cherry-picking.
A few beautiful examples will tempt overinterpretation. Mitigation: publish the full prompt list, seeds, generations, and scoring scripts. Show distributions, not only excerpts.

National and class bias.
“The future of Europe,” “a man’s duty,” “the King,” and “the Church” are not neutral across English-language corpora. They may overdraw British, Anglican, elite, masculine, or imperial registers. Mitigation: include alternate probes for family, town, work, school, law, neighbor, citizen, scientist, widow, child, and worker.

Fin-de-siècle fracture before 1914.
The prewar corpus may already contain decadence, pessimism, imperial anxiety, Nietzschean crisis, and anti-modern currents. The hypothesis should not assume simple prewar optimism. Mitigation: frame the expected result as a change in closure and inheritance, not a crude optimism/pessimism swing.

Postwar continuity.
Talkie-1930 may contain conservative, religious, imperial, or restorative texts. The postwar world did not become uniformly modernist. Mitigation: expect mixture. The signature should be distributional, not absolute.

7. DELIVERABLE FOR THE WRITEUP

If the probes work, the published artifact is a psychograph of model-inherited historical character: a set of figures showing that the Pre-WWI model, Talkie-1930, and modern reference occupy distinct regions across closure, duty, authority, progress, machine-modernity, and suffering. The paper should include anonymous representative completions, but the main claim should rest on convergent forced-choice logprob contrasts and blinded annotation. The claim would be modest but profound: a base model trained on a governed historical corpus can act as an instrument for detecting not merely the content of an age, but its habitual forms of moral and civilizational intelligibility.

If the probes fail, the artifact is still publishable: a null result in computational historical characterology. The writeup says that, under these corpora, scales, and probes, the hypothesized 1914 rupture is not separable from register, genre, tokenizer, and scale effects. That result would discipline the field before it becomes vibe science. It would also leave behind a reusable probe battery for future, larger, better genre-matched historical models.