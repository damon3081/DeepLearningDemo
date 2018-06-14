# coding:utf-8
import random

NUM_DIGITS = 3
MAX_GUSS = 10

def getSecretNum():
    # 返回一个由NUM_DIGITS 个不重复随机组成的字符串
    numbers = list(range(10))
    random.shuffle(numbers)
    secretNum = ''
    for i in range(NUM_DIGITS):
        secretNum += str(numbers[i])
    return secretNum

def getClues(guess, secretNum):
    # 返回一个由 Pico, Fermi 和Bagels 组成的，用来提示用户的字符串
    if guess == secretNum:
        return 'You got it!'
    clues = []
    for i in range(len(guess)):
        if guess[i] == secretNum[i]:
            clues.append('Fermi')
        elif guess[i] in secretNum:
            clues.append('Pico')
    if len(clues) == 0:
        return 'Bagels'
    clues.sort()
    return ' '.join(clues)

def isOnlydigits(num):
    # 如果字符串只包含数字，返回真，否则返回假
    if num == '':
        return False
    for i in num:
        if i not in '0 1 2 3 4 5 6 7 8 9'.split():
            return False
    return True

print('I am thinking of a %s-digit number. Try to guess what it is.' % (NUM_DIGITS))
print('The clues I give are...')
print('When I say:   Thant means:')
print('   Bagels    None of the digits is correct.')
print('   Pico      One digit is correct but in the wrong position.')
print('   Pico      One digit is correct but in the right position.')

while True:
    secretNum = getSecretNum()
    print('I have thought up a number. You have %s guesses to get it.' % (MAX_GUSS))
    guessesTaken = 1
    while guessesTaken <= MAX_GUSS:
        guess = ''
        while len(guess) != NUM_DIGITS or not isOnlydigits(guess):
            print('Guess #%s:  ' % guessesTaken)
            guess = input()
        print(getClues(guess, secretNum))
        guessesTaken += 1
        if guess == secretNum:
            break
        if guessesTaken > MAX_GUSS:
            print('Y ou ran out of guesses. The answer was %s.' % secretNum)
    print('Do you want to play again? (yes or no)')
    if not input().lower().startswith('y'):
        break
