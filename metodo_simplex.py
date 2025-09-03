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
            header += f"   x{i+1}  |"
        for i in range(m):
            header += f"   s{i+1}  |"
        header += "   RHS  |"
        print(header)
        print("-" * len(header))
        
        # Linhas das restrições
        for i in range(m):
            row = f" s{basic_vars[i]-n_original_vars+1}  |" if basic_vars[i] >= n_original_vars else f" x{basic_vars[i]+1}  |"
            for j in range(n-1):
                row += f" {tableau[i, j]:6.2f} |"
            row += f" {tableau[i, -1]:6.2f} |"
            print(row)
        
        # Linha Z
        print("-" * len(header))
        row = "  Z  |"
        for j in range(n-1):
            row += f" {tableau[-1, j]:6.2f} |"
        row += f" {tableau[-1, -1]:6.2f} |"
        print(row)
        print("="*80)
    
    def get_user_input(self):
        """Coleta os dados do problema do usuário."""
        print("\n" + "="*80)
        print("⊙ SOLUCIONADOR SIMPLEX INTERATIVO ⊙")
        print("="*80)
        print("\n📝 Este programa resolve problemas de Programação Linear no formato:")
        print("   MAXIMIZAR: Z = c₁x₁ + c₂x₂ + ... + cₙxₙ")
        print("   SUJEITO A: a₁₁x₁ + a₁₂x₂ + ... ≤ b₁")
        print("              a₂₁x₁ + a₂₂x₂ + ... ≤ b₂")
        print("              ...")
        print("              x₁, x₂, ... ≥ 0")
        
        # Número de variáveis e restrições
        print("\n🔢 DEFININDO O PROBLEMA:")
        n = int(input("   Quantas variáveis de decisão? "))
        m = int(input("   Quantas restrições? "))
        
        # Coeficientes da função objetivo
        print(f"\n📈 FUNÇÃO OBJETIVO (Maximizar Z):")
        print(f"   Digite os {n} coeficientes das variáveis:")
        C = []
        for i in range(n):
            C.append(float(input(f"   Coeficiente de x{i+1}: ")))
        C = np.array(C)
        
        # Matriz de restrições
        print(f"\n📋 RESTRIÇÕES:")
        A = []
        B = []
        for i in range(m):
            print(f"\n   Restrição {i+1}:")
            row = []
            for j in range(n):
                coef = float(input(f"     Coeficiente de x{j+1}: "))
                row.append(coef)
            A.append(row)
            B.append(float(input(f"     Lado direito (≤): ")))
        
        A = np.array(A)
        B = np.array(B)
        
        return A, B, C
    
    def simplex(self, A, B, C, show_steps=True):
        """
        Resolve problemas de programação linear pelo método Simplex com explicações.
        """
        self.iteration = 0
        m, n = A.shape
        
        print("\n" + "="*80)
        print("🚀 INICIANDO O MÉTODO SIMPLEX")
        print("="*80)
        
        # FASE 1: Construir a tabela inicial
        print("\n📌 FASE 1: CONSTRUÇÃO DO QUADRO INICIAL")
        print("   • Adicionando variáveis de folga para converter ≤ em =")
        print("   • Montando o quadro simplex inicial")
        
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
            
            # Verificar otimalidade
            if np.all(tableau[-1, :-1] >= 0):
                print("\n✅ SOLUÇÃO ÓTIMA ENCONTRADA!")
                print("   Todos os coeficientes da linha Z são não-negativos.")
                break
            
            # Escolher coluna pivô
            pivot_col = np.argmin(tableau[-1, :-1])
            print(f"\n🔍 Iteração {self.iteration}:")
            print(f"   • Coluna pivô: {pivot_col + 1} (menor valor negativo na linha Z: {tableau[-1, pivot_col]:.2f})")
            
            # Verificar se é ilimitado
            if np.all(tableau[:-1, pivot_col] <= 0):
                print("\n❌ PROBLEMA ILIMITADO!")
                print("   Todos os coeficientes na coluna pivô são não-positivos.")
                return {"status": "unbounded", "solution": None, "optimal_value": None}
            
            # Escolher linha pivô
            ratios = []
            print("\n   • Calculando razões (RHS / coef. coluna pivô):")
            for i in range(m):
                if tableau[i, pivot_col] > 0:
                    ratio = tableau[i, -1] / tableau[i, pivot_col]
                    ratios.append(ratio)
                    print(f"     Linha {i+1}: {tableau[i, -1]:.2f} / {tableau[i, pivot_col]:.2f} = {ratio:.2f}")
                else:
                    ratios.append(np.inf)
                    print(f"     Linha {i+1}: -- (coeficiente ≤ 0)")
            
            pivot_row = np.argmin(ratios)
            print(f"\n   • Linha pivô: {pivot_row + 1} (menor razão positiva: {ratios[pivot_row]:.2f})")
            print(f"   • Elemento pivô: {tableau[pivot_row, pivot_col]:.2f}")
            
            # Operação de pivoteamento
            print("\n   • Realizando operação de pivoteamento:")
            print(f"     1. Dividindo linha {pivot_row + 1} por {tableau[pivot_row, pivot_col]:.2f}")
            
            tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
            
            print("     2. Zerando outros elementos da coluna pivô")
            for i in range(m + 1):
                if i != pivot_row:
                    multiplier = tableau[i, pivot_col]
                    if abs(multiplier) > 0.001:  # Evitar imprimir operações triviais
                        print(f"        Linha {i+1 if i < m else 'Z'} -= {multiplier:.2f} × Linha {pivot_row + 1}")
                    tableau[i, :] -= multiplier * tableau[pivot_row, :]
            
            # Atualizar variáveis básicas
            old_var = basic_vars[pivot_row]
            basic_vars[pivot_row] = pivot_col
            
            var_out = f"s{old_var-n+1}" if old_var >= n else f"x{old_var+1}"
            var_in = f"x{pivot_col+1}" if pivot_col < n else f"s{pivot_col-n+1}"
            print(f"\n   • Troca de base: {var_out} sai, {var_in} entra")
            
            if show_steps:
                self.print_tableau(tableau, basic_vars, n, f"Após pivoteamento na linha {pivot_row+1}, coluna {pivot_col+1}")
                input("\n⏸️  Pressione Enter para continuar...")
        
        # FASE 3: Extrair solução
        print("\n📌 FASE 3: EXTRAÇÃO DA SOLUÇÃO")
        
        solution = np.zeros(n)
        print("\n   • Valores das variáveis de decisão:")
        for i, var in enumerate(basic_vars):
            if var < n:
                solution[var] = tableau[i, -1]
                print(f"     x{var+1} = {tableau[i, -1]:.2f} (variável básica)")
        
        for i in range(n):
            if i not in basic_vars:
                print(f"     x{i+1} = 0.00 (variável não-básica)")
        
        optimal_value = tableau[-1, -1]
        
        print(f"\n   • Valor ótimo da função objetivo: Z = {optimal_value:.2f}")
        
        return {"status": "optimal", "solution": solution, "optimal_value": optimal_value}
    
    def run_example(self):
        """Executa um exemplo pré-definido."""
        print("\n📚 EXEMPLO DEMONSTRATIVO")
        print("   Problema: Maximizar Z = 3x₁ + 2x₂")
        print("   Sujeito a: 2x₁ + x₂ ≤ 20")
        print("              x₁ + 2x₂ ≤ 20")
        print("              x₁ ≤ 10")
        print("              x₁, x₂ ≥ 0")
        
        A = np.array([[2, 1], [1, 2], [1, 0]])
        B = np.array([20, 20, 10])
        C = np.array([3, 2])
        
        return self.simplex(A, B, C)
    
    def verify_solution(self, A, B, C, solution):
        """Verifica se a solução satisfaz todas as restrições."""
        print("\n🔍 VERIFICAÇÃO DA SOLUÇÃO:")
        
        # Verificar restrições
        print("   • Verificando restrições:")
        for i in range(len(B)):
            lhs = np.dot(A[i], solution)
            print(f"     Restrição {i+1}: {lhs:.2f} ≤ {B[i]:.2f} ", end="")
            print("✓" if lhs <= B[i] + 0.0001 else "✗")
        
        # Verificar não-negatividade
        print("   • Verificando não-negatividade:")
        for i, val in enumerate(solution):
            print(f"     x{i+1} = {val:.2f} ≥ 0 ", end="")
            print("✓" if val >= -0.0001 else "✗")
        
        # Calcular valor da função objetivo
        z_value = np.dot(C, solution)
        print(f"\n   • Valor da função objetivo: Z = {z_value:.2f}")


def main():
    solver = SimplexSolver()
    
    while True:
        print("\n" + "="*80)
        print("📊 MENU PRINCIPAL")
        print("="*80)
        print("1. Resolver novo problema")
        print("2. Executar exemplo demonstrativo")
        print("3. Sair")
        
        choice = input("\nEscolha uma opção (1-3): ")
        
        if choice == '1':
            try:
                A, B, C = solver.get_user_input()
                result = solver.simplex(A, B, C)
                
                if result["status"] == "optimal":
                    print("\n" + "="*80)
                    print("🎉 RESUMO DA SOLUÇÃO ÓTIMA")
                    print("="*80)
                    print(f"Status: {result['status']}")
                    print(f"Solução ótima: {result['solution']}")
                    print(f"Valor ótimo: {result['optimal_value']:.2f}")
                    
                    solver.verify_solution(A, B, C, result['solution'])
                
            except ValueError:
                print("\n❌ Erro: Por favor, insira valores numéricos válidos.")
            except Exception as e:
                print(f"\n❌ Erro inesperado: {e}")
        
        elif choice == '2':
            result = solver.run_example()
            if result["status"] == "optimal":
                A = np.array([[2, 1], [1, 2], [1, 0]])
                B = np.array([20, 20, 10])
                C = np.array([3, 2])
                solver.verify_solution(A, B, C, result['solution'])
        
        elif choice == '3':
            print("\n👋 Obrigado por usar o Solucionador Simplex!")
            break
        
        else:
            print("\n❌ Opção inválida. Por favor, escolha 1, 2 ou 3.")


if __name__ == "__main__":
    main()
