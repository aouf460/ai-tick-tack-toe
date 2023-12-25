import random
import numpy as np
import pickle

def print_board(board):
    for row in board:
        print(" ".join(row))
    print()

def is_winner(board, player):
    for row in board:
        if all(cell == player for cell in row):
            return True

    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True

    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True

    return False

def is_board_full(board):
    return all(all(cell != ' ' for cell in row) for row in board)

def get_user_move(board):
    while True:
        try:
            row = int(input("Enter the row (0, 1, or 2): "))
            col = int(input("Enter the column (0, 1, or 2): "))

            if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == ' ':
                return row, col
            else:
                print("Invalid move. Try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_computer_move(board, q_table, epsilon):
    if np.random.rand() < epsilon:
        return random.choice([(row, col) for row in range(3) for col in range(3) if board[row][col] == ' '])
    else:
        state = tuple(tuple(row) for row in board)
        available_actions = [(row, col) for row in range(3) for col in range(3) if board[row][col] == ' ']
        q_values = [q_table.get((state, action), 0) for action in available_actions]
        best_actions = [action for action, q_value in zip(available_actions, q_values) if q_value == max(q_values)]
        return random.choice(best_actions)

def update_q_table(q_table, state, action, reward, next_state):
    current_q = q_table.get((state, action), 0)
    max_future_q = max(q_table.get((next_state, a), 0) for a in [(i, j) for i in range(3) for j in range(3)])
    new_q = current_q + learning_rate * (reward + discount_factor * max_future_q - current_q)
    q_table[(state, action)] = new_q

def save_q_table(q_table, filename='q_table.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(q_table, file)

def load_q_table(filename='q_table.pkl'):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return {}

def play_game(q_table, epsilon):
    board = [[' ' for _ in range(3)] for _ in range(3)]
    players = ['X', 'O']
    current_player = random.choice(players)

    while True:
        print_board(board)

        if current_player == 'X':
            row, col = get_user_move(board)
        else:
            print("Computer's move:")
            row, col = get_computer_move(board, q_table, epsilon)
            print(f"Row: {row}, Column: {col}")

        state = tuple(tuple(row) for row in board)
        action = (row, col)
        board[row][col] = current_player

        if is_winner(board, current_player):
            print_board(board)
            if current_player == 'X':
                print("Congratulations! You win!")
                update_q_table(q_table, state, action, 1, None)
            else:
                print("Computer wins!")
                update_q_table(q_table, state, action, -1, None)
            break
        elif is_board_full(board):
            print_board(board)
            print("It's a tie!")
            update_q_table(q_table, state, action, 0, None)
            break

        current_player = 'X' if current_player == 'O' else 'O'

if __name__ == "__main__":
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.1
    q_table = load_q_table()

    for _ in range(1000):  # Train the computer with 1000 games
        play_game(q_table, epsilon)

    save_q_table(q_table)  # Save the learned strategy

    play_game(q_table, 0)  # Play a game with the learned strategy
