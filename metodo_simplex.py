import numpy as np
import sys

class SimplexSolver:
    def __init__(self):
        self.iteration = 0
        
    def print_tableau(self, tableau, basic_vars, n_original_vars, iteration_msg=""):
        """Imprime o quadro simplex de forma formatada."""
        m, n = tableau.shape
        m -= 1  # Excluir linha da função objetivo
        
        print(f"\n{'='*80}")
        if iteration_msg:
            print(f"🔄 {iteration_msg}")
        print(f"📊 QUADRO SIMPLEX - Iteração {self.iteration}")
        print(f"{'='*80}")
        
        # Cabeçalho
        header = "Base |"
        for i in range(n_original_vars):
            header += f"   x{i+1}   |"
        for i in range(m):
            header += f"   s{i+1}   |"
        header += "    RHS   |"
        print(header)
        print("-" * len(header))
        
        # Linhas das restrições
        for i in range(m):
            # Constrói o nome da variável na base
            if basic_vars[i] >= n_original_vars:
                var_name = f"s{basic_vars[i] - n_original_vars + 1}"
            else:
                var_name = f"x{basic_vars[i] + 1}"
            
            row = f" {var_name:<4}|" # Alinha o nome da variável

            for j in range(n-1):
                row += f" {tableau[i, j]:6.2f} |"
            row += f" {tableau[i, -1]:8.2f} |"
            print(row)
        
        # Linha Z
        print("-" * len(header))
        row = "  Z   |"
        for j in range(n-1):
            row += f" {tableau[-1, j]:6.2f} |"
        row += f" {tableau[-1, -1]:8.2f} |"
        print(row)
        print("="*80)
    
    def get_user_input(self):
        """Coleta os dados do problema do usuário."""
        print("\n" + "="*80)
        print("⊙ SOLUCIONADOR SIMPLEX INTERATIVO ⊙")
        print("="*80)

        problem_type = ''
        while problem_type not in ['1', '2']:
            print("\n🎯 TIPO DO PROBLEMA:")
            print("  1. Maximização (com restrições ≤)")
            print("  2. Minimização (com restrições ≥)")
            problem_type = input("  Escolha o tipo de problema (1 ou 2): ")
            if problem_type not in ['1', '2']:
                print("  ❌ Opção inválida. Por favor, digite 1 ou 2.")
        
        problem_type = 'max' if problem_type == '1' else 'min'

        print("\n📝 Este programa resolve problemas de Programação Linear no formato:")
        if problem_type == 'max':
            print("   MAXIMIZAR: Z = c₁x₁ + c₂x₂ + ... + cₙxₙ")
            print("   SUJEITO A: a₁₁x₁ + a₁₂x₂ + ... ≤ b₁")
        else:
            print("   MINIMIZAR: Z = c₁x₁ + c₂x₂ + ... + cₙxₙ")
            print("   SUJEITO A: a₁₁x₁ + a₁₂x₂ + ... ≥ b₁")
        
        print("             ...")
        print("             x₁, x₂, ... ≥ 0")
        
        # Número de variáveis e restrições
        print("\n🔢 DEFININDO O PROBLEMA:")
        n = int(input("  Quantas variáveis de decisão? "))
        m = int(input("  Quantas restrições? "))
        
        # Coeficientes da função objetivo
        print(f"\n📈 FUNÇÃO OBJETIVO ({'Maximizar' if problem_type == 'max' else 'Minimizar'} Z):")
        print(f"  Digite os {n} coeficientes das variáveis:")
        C = []
        for i in range(n):
            C.append(float(input(f"  Coeficiente de x{i+1}: ")))
        C = np.array(C)
        
        # Matriz de restrições
        print(f"\n📋 RESTRIÇÕES:")
        A = []
        B = []
        constraint_symbol = '≤' if problem_type == 'max' else '≥'
        for i in range(m):
            print(f"\n  Restrição {i+1}:")
            row = []
            for j in range(n):
                coef = float(input(f"    Coeficiente de x{j+1}: "))
                row.append(coef)
            A.append(row)
            B.append(float(input(f"    Lado direito ({constraint_symbol}): ")))
        
        A = np.array(A)
        B = np.array(B)
        
        return A, B, C, problem_type
    
    def simplex(self, A, B, C, problem_type='max', show_steps=True):
        """
        Resolve problemas de programação linear pelo método Simplex com explicações.
        """
        self.iteration = 0
        m, n = A.shape
        
        print("\n" + "="*80)
        print("🚀 INICIANDO O MÉTODO SIMPLEX")
        print("="*80)

        if problem_type == 'min':
            # Converte o problema de minimização para um de maximização equivalente
            # 1. min Z  ->  max Z' = -Z
            C = -C
            # 2. A*x >= B  -> -A*x <= -B
            A = -A
            B = -B
            print("   Convertendo para um problema de Maximização (≤) equivalente:")
            print("   1. Função Objetivo: Coeficientes multiplicados por -1.")
            print("   2. Restrições: Coeficientes e RHS multiplicados por -1.")

        # FASE 1: Construir a tabela inicial
        print("\n📌 FASE 1: CONSTRUÇÃO DO QUADRO INICIAL")
        print("  • Adicionando variáveis de folga para converter ≤ em =")
        print("  • Montando o quadro simplex inicial")
        
        tableau = np.zeros((m + 1, n + m + 1))
        tableau[:m, :n] = A
        tableau[:m, n:n + m] = np.eye(m)
        tableau[:m, -1] = B
        tableau[-1, :n] = -C
        
        basic_vars = list(range(n, n + m))
        
        if show_steps:
            self.print_tableau(tableau, basic_vars, n, "Quadro inicial com variáveis de folga")
            input("\n⏸️  Pressione Enter para continuar...")
        
        # FASE 2: Iterações do Simplex
        print("\n📌 FASE 2: ITERAÇÕES DO MÉTODO SIMPLEX")
        
        while True:
            self.iteration += 1
            
            # Verificar otimalidade (tolerância para erros de ponto flutuante)
            if np.all(tableau[-1, :-1] >= -1e-9):
                print("\n✅ SOLUÇÃO ÓTIMA ENCONTRADA!")
                print("  Todos os coeficientes da linha Z são não-negativos.")
                break
            
            # Escolher coluna pivô
            pivot_col = np.argmin(tableau[-1, :-1])
            print(f"\n🔍 Iteração {self.iteration}:")
            print(f"  • Coluna pivô: {pivot_col + 1} (menor valor negativo na linha Z: {tableau[-1, pivot_col]:.2f})")
            
            # Verificar se é ilimitado
            if np.all(tableau[:-1, pivot_col] <= 1e-9):
                print("\n❌ PROBLEMA ILIMITADO!")
                print("  Todos os coeficientes na coluna pivô são não-positivos.")
                return {"status": "unbounded", "solution": None, "optimal_value": None}
            
            # Escolher linha pivô
            ratios = []
            print("\n  • Calculando razões (RHS / coef. coluna pivô):")
            for i in range(m):
                if tableau[i, pivot_col] > 1e-9:
                    ratio = tableau[i, -1] / tableau[i, pivot_col]
                    ratios.append(ratio)
                    print(f"    Linha {i+1}: {tableau[i, -1]:.2f} / {tableau[i, pivot_col]:.2f} = {ratio:.2f}")
                else:
                    ratios.append(np.inf)
                    print(f"    Linha {i+1}: -- (coeficiente ≤ 0)")
            
            pivot_row = np.argmin(ratios)
            print(f"\n  • Linha pivô: {pivot_row + 1} (menor razão positiva: {ratios[pivot_row]:.2f})")
            pivot_element = tableau[pivot_row, pivot_col]
            print(f"  • Elemento pivô: {pivot_element:.2f}")
            
            # Operação de pivoteamento
            print("\n  • Realizando operação de pivoteamento:")
            print(f"    1. Normalizando a linha pivô (Linha {pivot_row + 1} /= {pivot_element:.2f})")
            tableau[pivot_row, :] /= pivot_element
            
            print("    2. Zerando outros elementos da coluna pivô")
            for i in range(m + 1):
                if i != pivot_row:
                    multiplier = tableau[i, pivot_col]
                    if abs(multiplier) > 1e-9:
                        print(f"       Linha {i+1 if i < m else 'Z'} -= {multiplier:.2f} × (Nova Linha {pivot_row + 1})")
                    tableau[i, :] -= multiplier * tableau[pivot_row, :]
            
            # Atualizar variáveis básicas
            basic_vars[pivot_row] = pivot_col
            
            if show_steps:
                self.print_tableau(tableau, basic_vars, n, f"Após pivoteamento na linha {pivot_row+1}, coluna {pivot_col+1}")
                input("\n⏸️  Pressione Enter para continuar...")
        
        # FASE 3: Extrair solução
        print("\n📌 FASE 3: EXTRAÇÃO DA SOLUÇÃO")
        
        solution = np.zeros(n)
        print("\n  • Valores das variáveis de decisão:")
        for i, var_index in enumerate(basic_vars):
            if var_index < n:
                solution[var_index] = tableau[i, -1]
                print(f"    x{var_index+1} = {tableau[i, -1]:.2f} (variável básica)")
        
        for i in range(n):
            if i not in basic_vars:
                print(f"    x{i+1} = 0.00 (variável não-básica)")
        
        optimal_value = tableau[-1, -1]

        if problem_type == 'min':
            optimal_value = -optimal_value
            print(f"\n  • Valor ótimo da função convertida (Z'): {tableau[-1, -1]:.2f}")
            print(f"  • Valor MÍNIMO final (Z = -Z'): {optimal_value:.2f}")
        else:
            print(f"\n  • Valor MÁXIMO da função objetivo: Z = {optimal_value:.2f}")
        
        return {"status": "optimal", "solution": solution, "optimal_value": optimal_value}
    
    def run_example(self):
        """Executa um exemplo pré-definido."""
        print("\n📚 EXEMPLO DEMONSTRATIVO (MAXIMIZAÇÃO)")
        print("  Problema: Maximizar Z = 3x₁ + 2x₂")
        print("  Sujeito a: 2x₁ + x₂ ≤ 20")
        print("             x₁ + 2x₂ ≤ 20")
        print("             x₁, x₂ ≥ 0")
        
        A = np.array([[2, 1], [1, 2]])
        B = np.array([20, 20])
        C = np.array([3, 2])
        
        return self.simplex(A, B, C, problem_type='max')
    
    def verify_solution(self, A, B, C, solution, problem_type):
        """Verifica se a solução satisfaz todas as restrições originais."""
        print("\n🔍 VERIFICAÇÃO DA SOLUÇÃO (com dados originais):")
        
        constraint_symbol = '≤' if problem_type == 'max' else '≥'
        print(f"  • Verificando restrições ({constraint_symbol}):")
        
        for i in range(len(B)):
            lhs = np.dot(A[i], solution)
            if problem_type == 'max':
                is_ok = lhs <= B[i] + 1e-5
            else: # min
                is_ok = lhs >= B[i] - 1e-5
            
            print(f"    Restrição {i+1}: {lhs:.2f} {constraint_symbol} {B[i]:.2f} ", end="")
            print("✓" if is_ok else "✗")
        
        print("  • Verificando não-negatividade:")
        for i, val in enumerate(solution):
            is_ok = val >= -1e-5
            print(f"    x{i+1} = {val:.2f} ≥ 0 ", end="")
            print("✓" if is_ok else "✗")
        
        z_value = np.dot(C, solution)
        print(f"\n  • Valor da função objetivo com a solução encontrada: Z = {z_value:.2f}")


def main():
    solver = SimplexSolver()
    
    while True:
        print("\n" + "="*80)
        print("📊 MENU PRINCIPAL")
        print("="*80)
        print("1. Resolver novo problema")
        print("2. Executar exemplo demonstrativo (Maximização)")
        print("3. Sair")
        
        choice = input("\nEscolha uma opção (1-3): ")
        
        if choice == '1':
            try:
                A, B, C, problem_type = solver.get_user_input()
                # Salva os dados originais para a verificação final
                original_A, original_B, original_C = A.copy(), B.copy(), C.copy()
                
                result = solver.simplex(A, B, C, problem_type)
                
                if result["status"] == "optimal":
                    print("\n" + "="*80)
                    print("🎉 RESUMO DA SOLUÇÃO ÓTIMA")
                    print("="*80)
                    solution_str = ", ".join([f"x{i+1}={val:.2f}" for i, val in enumerate(result['solution'])])
                    print(f"Solução ótima: [ {solution_str} ]")
                    print(f"Valor {'Mínimo' if problem_type == 'min' else 'Máximo'}: Z = {result['optimal_value']:.2f}")
                    
                    solver.verify_solution(original_A, original_B, original_C, result['solution'], problem_type)
            
            except ValueError:
                print("\n❌ Erro: Por favor, insira valores numéricos válidos.")
            except Exception as e:
                print(f"\n❌ Erro inesperado: {e}")
        
        elif choice == '2':
            result = solver.run_example()
            if result["status"] == "optimal":
                A = np.array([[2, 1], [1, 2]])
                B = np.array([20, 20])
                C = np.array([3, 2])
                solver.verify_solution(A, B, C, result['solution'], 'max')
        
        elif choice == '3':
            print("\n👋 Obrigado por usar o Solucionador Simplex!")
            break
        
        else:
            print("\n❌ Opção inválida. Por favor, escolha 1, 2 ou 3.")

if __name__ == "__main__":
    main()
