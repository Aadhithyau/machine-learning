import math

def finalResult(x, aAnswer, rAnswer):
    print(f"{x:<10} {aAnswer:<15.2f} {rAnswer:<5.2f}")

def mean(a, r, l):
    aResultMean = a / l
    rResultMean = r / l
    finalResult("mean", aResultMean, rResultMean)

def count(aCount, rCount):
    finalResult("count", aCount, rCount)

def minAndMax(aArray, rArray):
    aSorted = sorted(aArray)
    rSorted = sorted(rArray)
    maxAgeArray = aSorted[-1]
    maxRatingArray = rSorted[-1]
    minAgeArray = aSorted[0]
    minRatingArray = rSorted[0]
    finalResult("min", minAgeArray, minRatingArray)
    finalResult("max", maxAgeArray, maxRatingArray)

def std(ageArray, ratingArray, aSummation, rSummation):
    aSummation = aSummation / len(ageArray)
    rSummation = rSummation / len(ratingArray)
    aAnswer = 0.0
    rAnswer = 0.0
    for i in range(len(ageArray)):
        aAnswer += (ageArray[i] - aSummation) ** 2
        rAnswer += (ratingArray[i] - rSummation) ** 2
    aAnswer = math.sqrt(aAnswer / (len(ageArray) - 1))
    rAnswer = math.sqrt(rAnswer / (len(ratingArray) - 1))
    finalResult("std", aAnswer, rAnswer)

# Main program
name = ["Tom", "James", "Ricky", "Vin", "Steve", "Smith", "Jack", "Lee", "David", "Gasper", "Betina", "Andres"]
age = [25, 26, 25, 23, 30, 29, 23, 34, 40, 30, 51, 46]
rating = [4.23, 3.24, 3.98, 2.56, 3.20, 4.6, 3.8, 3.78, 2.98, 4.80, 4.10, 3.65]

print(f"{'NAME':<10} {'AGE':<15} {'RATING':<5}")
print(f"{'---':<10} {'----':<15} {'---':<5}")

asumfx = 0
rsumfx = 0
for i in range(len(name)):
    print(f"{name[i]:<10} {age[i]:<15} {rating[i]:<5.2f}")
    asumfx += age[i]
    rsumfx += rating[i]

print()
print(f"{'DESCRIBE':<10} {'AGE':<15} {'RATING':<5}")
print(f"{'---':<10} {'----':<15} {'---':<5}")

mean(asumfx, rsumfx, len(name))
minAndMax(age, rating)
count(len(age), len(rating))
std(age, rating, asumfx, rsumfx)
