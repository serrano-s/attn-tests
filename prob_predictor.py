probs_of_uninterp = [0.0871252488287455, 0.17563790642028432, 0.653690708382764, 0.790742565298695]  # yahoo
total = 49733  # yahoo
probs_of_uninterp = [0.45549544208107906, 0.5309419699103238, 0.6451493366931002, 0.5466538204995183]  # imdb
total = 13493  # imdb
probs_of_uninterp = [0.24615796937231565, 0.35901867922351965, 0.7142445193814755, 0.5339150376909142]  # amazon
total = 589532  # amazon
probs_of_uninterp = [0.31526378871669786, 0.47784763546901615, 0.7055743633955044, 0.6329667556826545]  # yelp
total = 47557  # yelp


def get_predicted_probabilities(p1, p2, p3, p4):
    prob_all_4 = p1 * p2 * p3 * p4
    prob_exactly_3 = ((1 - p1) * p2 * p3 * p4) + (p1 * (1 - p2) * p3 * p4) + (p1 * p2 * (1 - p3) * p4) + \
                     (p1 * p2 * p3 * (1 - p4))
    list_of_probs = [p1, p2, p3, p4]
    prob_exactly_2 = 0
    for i in range(4):
        for j in range(4):
            if j <= i:
                continue
            other_inds = {0:0, 1:1, 2:2, 3:3}
            del other_inds[i]
            del other_inds[j]
            other_inds = list(other_inds.keys())
            prob_exactly_2 += (list_of_probs[i] * list_of_probs[j] * (1 - list_of_probs[other_inds[0]]) *
                               (1 - list_of_probs[other_inds[1]]))
    prob_exactly_1 = (p1 * (1 - p2) * (1 - p3) * (1 - p4)) + ((1 - p1) * p2 * (1 - p3) * (1 - p4)) + \
    ((1 - p1) * (1 - p2) * p3 * (1 - p4)) + ((1 - p1) * (1 - p2) * (1 - p3) * p4)
    prob_exactly_0 = ((1 - p1) * (1 - p2) * (1 - p3) * (1 - p4))
    return prob_all_4, prob_exactly_3, prob_exactly_2, prob_exactly_1, prob_exactly_0


prob_all_4, prob_exactly_3, prob_exactly_2, prob_exactly_1, prob_exactly_0 =\
    get_predicted_probabilities(probs_of_uninterp[0], probs_of_uninterp[1], probs_of_uninterp[2], probs_of_uninterp[3])

print("\tProbs sum to " + str(prob_all_4 + prob_exactly_0 + prob_exactly_1 + prob_exactly_2 + prob_exactly_3))
print()

print("Prob all 4: " + str(prob_all_4))
print("Prob exactly 3: " + str(prob_exactly_3))
print("Prob exactly 2: " + str(prob_exactly_2))
print("Prob exactly 1: " + str(prob_exactly_1))
print("Prob never uninterpretable: " + str(prob_exactly_0))

print()

print("Num expected 4: " + str(prob_all_4 * total))
print("Num expected 3: " + str(prob_exactly_3 * total))
print("Num expected 2: " + str(prob_exactly_2 * total))
print("Num expected 1: " + str(prob_exactly_1 * total))
print("Num expected 0: " + str(prob_exactly_0 * total))
