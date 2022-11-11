# MINIPROJETO - OTIMIZAÇÃO PELO VETOR GRADIENTE
### Link do Github:
* https://github.com/PedroPertusi/Maximun-Minimum-MultivariableFunctions.git
### ALUNOS:
* JOÃO LUCAS CADORNIGA
* PEDRO VAZ DE MORAES PERTUSI

# Tarefa 1
### Considere a função f dada pela lei: &nbsp;&nbsp;&nbsp; $ f(x,y) = 3x^2 + 3xy + 2y^2 + x + y$ <br>
### Vetor Gradiente de Ponto Genérico: &nbsp;&nbsp;&nbsp; $ \nabla f(x,y) = (6x + 3y + 1;3x + 4y + 1) $
<br>
### Gráfico no GeoGebra:
<img src="Captura de Tela 2022-11-11 às 16.20.07.png"/>

## Função que Calcula o Ponto Máximo da Função:



```python
def maximize_2var_func(x0, y0, f, grad, alpha, precision):
    delx, dely = grad(x0, y0)
    x1 = x0 + alpha*delx
    y1 = y0 + alpha*dely

    iteracoes = 1
    while abs(x1 - x0) > precision and abs(y1 - y0) > precision:
        x0 = x1
        y0 = y1
        delx, dely = grad(x0, y0)
        x1 = x0 + alpha*delx
        y1 = y0 + alpha*dely
        iteracoes += 1
    
    return x0, y0, f(x0, y0), iteracoes
```

## Função que Calcula o Ponto Mínimo da Função:


```python
def minimize_2var_func(x0, y0, f, grad, alpha, precision):
    delx, dely = grad(x0, y0)
    x1 = x0 + alpha*-1*delx
    y1 = y0 + alpha*-1*dely

    iteracoes = 1
    while abs(x1 - x0) > precision and abs(y1 - y0) > precision:
        x0 = x1
        y0 = y1
        delx, dely = grad(x0, y0)
        x1 = x0 + alpha*-1*delx
        y1 = y0 + alpha*-1*dely
        iteracoes += 1
    
    return x0, y0, f(x0, y0), iteracoes
```

### Função e Gradiente em Python


```python
def gradient(x, y):
    return [6*x + 3*y + 1, 3*x + 4*y + 1]

def function(x, y):
    return (3 * (x**2)) + (3 * x * y) + (2 * (y**2)) + x + y
```

### c) Achando ponto de mínimo para | $\alpha$ = 0,1 | Ponto Inicial(Xo,Yo) = (0,0) | Precisão 10<sup>-5</sup>


```python
x,y,z,interacoes = minimize_2var_func(0, 0, function, gradient, 0.1, 10**(-5))
print(f'Valor do Mínimo de f(x): ({x}; {y}; {z}) - Em {interacoes} Interações')
```

    Valor do Mínimo de f(x): (-0.06671486137110971; -0.19993313341945276; -0.1333333270907108) - Em 37 Interações


### d) Repetindo o processo do ponto de mínimo para | $\alpha$ = 0,15 | $\alpha$ = 0,2 | $\alpha$ = 0,3 | $\alpha$ = 0,5 | <br>

##### Abaixo repetimos o procedimento aumentando o $\alpha$, logo aumentando os passos utilizados. Com isso, percebemos que para um $\alpha$ grande isso leva o programa a divergir muito do minimo, fugindo assim cada vez mas do valor, no computador levando a um erro de overflow (estouro na memória) por conta de fugir infinitamente do mínimo. De forma análoga, um $\alpha$ muito pequeno pode levar o nosso programa demorar muitas interacoes para chegar no mínimo esperado, apesar de conseguir assim uma precisão maior.



```python
from math import inf
x,y,z,interacoes = minimize_2var_func(0, 0, function, gradient, 0.15, 10**(-5))
print(f'Valor do Mínimo de f(x): ({x}; {y}; {z}) - Em {interacoes} Interações')
x,y,z,interacoes = minimize_2var_func(0, 0, function, gradient, 0.2, 10**(-5))
print(f'Valor do Mínimo de f(x): ({x}; {y}; {z}) - Em {interacoes} Interações')
# x,y,z,interacoes = minimize_2var_func(0, 0, function, gradient, 0.3, 10**(-5)) --> Erro 
print(f'Valor do Mínimo de f(x): ({inf}; {inf}; erro) - Em {inf} Interações')
#x,y,z,interacoes = minimize_2var_func(0, 0, function, gradient, 0.5, 10**(-5)) --> Erro
print(f'Valor do Mínimo de f(x): ({inf}; {inf}; erro) - Em {inf} Interações')
```

    Valor do Mínimo de f(x): (-0.06669803079239797; -0.1999564846000459; -0.13333333068949554) - Em 25 Interações
    Valor do Mínimo de f(x): (-0.06668064768000002; -0.2; -0.1333333327469271) - Em 22 Interações
    Valor do Mínimo de f(x): (inf; inf; erro) - Em inf Interações
    Valor do Mínimo de f(x): (inf; inf; erro) - Em inf Interações


# Tarefa 2
### Considere a função f dada pela lei: &nbsp;&nbsp;&nbsp; $g(x,y) = \sqrt{ x^2 + y^2 + 1 } + x^2 ℯ^{-y^2} + (x-2)^2$ <br>
### Vetor Gradiente de Ponto Genérico: &nbsp;&nbsp;&nbsp; $ ∇g(x,y) = (x(1/\sqrt{x^2 + y^2 + 1} -2x^2e^{-y^2} + 2) - 4 ; y/\sqrt{x^2+y^2+1} -2x^2ye^{-yˆ2})$  <br>
### Gráfico no GeoGebra:
<img src="Captura de Tela 2022-11-11 às 16.19.31.png"/>


### Função e Gradiente em Python


```python
import numpy as np

def gradient(x, y):
    delx = x * (1/np.sqrt(x**2 + y**2 + 1) + 2 * np.e**(-(y**2)) + 2) - 4
    dely = y/np.sqrt(x**2 + y**2 + 1) - 2 * x**2 * np.e**(-(y**2))*y
    return [delx, dely]

def function(x, y):
    return np.sqrt(x**2 + y**2 + 1) + (x**2 * np.e**(-(y**2))) + (x - 2)**2
```

### Achando ponto de mínimo para | $\alpha$ = 0,1 | Ponto Inicial(Xo,Yo) = (-0.1,-0.1) | Precisão 10<sup>-5</sup>
##### Para obter o ponto de mínimo foi necessário usar valores iniciais diferentes, isso correu por conta de o g(0,0) ser exatamente o ponto médio entre os dois mínimos, fazendo assim com que seja necessário pegar pontos difetentes do (0,0) para obter assim os dois mínimos (como mostrado abaixo).


```python
x,y,z,interacoes = minimize_2var_func(-0.1, -0.1, function, gradient, 0.1, 10**(-5))
print(f'Valor do Mínimo de f(x): ({x}; {y}; {z}) - Em {interacoes} Interações')
```

    Valor do Mínimo de f(x): (1.5460972463784461; -1.5641379893241454; 2.828999745276761) - Em 125 Interações


### Achando ponto de mínimo para | $\alpha$ = 0,1 | Ponto Inicial(Xo,Yo) = (0.1,0.1) | Precisão 10<sup>-5</sup>


```python
x,y,z,interacoes = minimize_2var_func(0.1, 0.1, function, gradient, 0.1, 10**(-5))
print(f'Valor do Mínimo de f(x): ({x}; {y}; {z}) - Em {interacoes} Interações')
```

    Valor do Mínimo de f(x): (1.5460961680122989; 1.5641365673822754; 2.8289997455553983) - Em 124 Interações


# Tarefa 3
#### Considere a função f dada pela lei: &nbsp;&nbsp;&nbsp; $h(x,y) = 4e^{-x^2-y^2} + 3e^{-x^2-y^2+4x+6y-13} - \frac{x^2}{4} - \frac{y^2}{6} +2 $ <br>
#### Vetor Gradiente de Ponto Genérico: &nbsp;&nbsp;&nbsp; $ ∇h(x,y) = (-6(x-2)e^{-x^2+4x-y^2+6y-13} -8xe^{-x^2-y^2} - \frac{x}{2};-6(y-3)e^{-x^2+4x-y^2+6y-13} -8ye^{-x^2-y^2} -\frac{y}{3})$  <br>
<img src="Captura de Tela 2022-11-11 às 16.18.34.png"/>

### Função e Gradiente em Python 


```python
def function(x,y):
    return 4*np.e**(-(x**2)-(y**2)) + 3*np.e**(-(x**2)-(y**2)+4*x+6*y-13) - ((x**2)/4) - ((y**2)/6) + 2

def gradient(x, y):
    delx = -6 * (x-2) * np.e ** (-(x**2)+ 4*x - (y**2) + 6*y - 13) - 8 * x * np.e ** (-(x**2)-(y**2)) - x/2
    dely = -6 * (y-3) * np.e ** (-(x**2)+ 4*x - (y**2) + 6*y - 13) - 8 * y * np.e ** (-(x**2)-(y**2)) - y/3
    return [delx, dely]
```

### Achando ponto de mínimo para | $\alpha$ = 0,1 | Ponto Inicial(Xo,Yo) = (1,1) | Precisão 10<sup>-5</sup>


```python
x,y,z,interacoes = maximize_2var_func(1, 1, function, gradient, 0.1, 10**(-5))
print(f'Valor do Mínimo de f(x): ({x}; {y}; {z}) - Em {interacoes} Interações')
```

    Valor do Mínimo de f(x): (5.511506203443253e-06; 9.679233843170952e-06; 6.000006781012079) - Em 11 Interações


### Achando ponto de mínimo para | $\alpha$ = 0,1 | Ponto Inicial(Xo,Yo) = (4,4) | Precisão 10<sup>-5</sup>


```python
x,y,z,interacoes = maximize_2var_func(4, 4, function, gradient, 0.1, 10**(-5))
print(f'Valor do Mínimo de f(x): ({x}; {y}; {z}) - Em {interacoes} Interações')
```

    Valor do Mínimo de f(x): (1.8383261266233608; 2.8338353568997423; 2.659755914118903) - Em 20 Interações


# Tarefa Desafio - Passo Variável

Para que possamos melhorar a eficiência das nossas funções maximizadoras e minimizadoras, precisamos utilizar os próprios parâmetros $x_0$, $y_0$, e as derivadas parciais da função, já que um passo fixo não nos permite adaptar a nossa mudança de acordo com a distância que estamos do ponto de máximo.

Ao traçarmos o vetor gradiente no plano $xy$, criamos um plano vertical que intersecta a superfície gerada pela função $f$, formando uma curva. O ponto máximo dessa curva deve ser o nosso $\alpha$, já que, portanto, será o mais próximo do máximo de $f$.

Assim, para realizar esse cálculo, precisamos primeiro do gradiente (da função da tarefa 1) em $(x_0, y_0)$:

$\nabla f(x_0, y_0) = (6x_0 + 3y_0 + 1; 3x_0 + 4y_0 + 1)$

Vamos chamar esses valores de $(a, b)$.

Agora, considerando que $x_1 = x_0 + \alpha * a$ e $y_1 = y_0 + \alpha * b$, substituimos esses valores em $f$:

$f(x_1, y_1) = 3(x_0 + \alpha * a)² + 3(x_0 + \alpha * a)(y_0 + \alpha * b) + 2(y_0 + \alpha * b)² + (x_0 + \alpha * a) + (y_0 + \alpha * b)$  

Acabamos de obter $f(\alpha)$! Como queremos maximizá-la, sua derivada deve ser 0:

$f'(\alpha) = 6 \alpha a² + a + 6ab \alpha + 6ax_0 + 3ay_0 + 4 \alpha b² + b + 3bx_0 + 4by_0 = 0$

Isolando $\alpha$, encontramos:

$\alpha = \frac{-6ax_0 - 3ay_0 - a - 3bx_0 - 4by_0 - b}{2(3a² + 3ab = 2b²)}$

Tal que $a$ e $b$ são, respectivamente, as derivadas parciais em função de x e de y.

Agora, vamos remontar as nossas funções, calculando $\alpha$ a cada iteração. Perceba que o número de passos diminui de **37 para 9**.


```python
def alpha(x0, y0, a, b):
    num = -6*a*x0 + (-3*a*y0) - a + (-3*b*x0) + (-4*b*y0) - b
    den = 2*(3*(a**2) + 3*a*b + 2*(b**2))
    return abs(num/den)


def minimize_2var_func_V2(x0, y0, f, grad, precision):
    delx, dely = grad(x0, y0)
    a = alpha(x0, y0, delx, dely)
    x1 = x0 + a*-1*delx
    y1 = y0 + a*-1*dely

    iteracoes = 1
    while abs(x1 - x0) > precision and abs(y1 - y0) > precision:
        x0 = x1
        y0 = y1
        delx, dely = grad(x0, y0)
        a = alpha(x0, y0, delx, dely)
        print(f'Alpha {iteracoes}: {a}')
        x1 = x0 + a*-1*delx
        y1 = y0 + a*-1*dely
        iteracoes += 1
    
    return x0, y0, f(x0, y0), iteracoes



def maximize_2var_func_V2(x0, y0, f, grad, precision):
    delx, dely = grad(x0, y0)
    a = alpha(x0, y0, delx, dely)
    x1 = x0 + a*delx
    y1 = y0 + a*dely

    iteracoes = 1
    while abs(x1 - x0) > precision and abs(y1 - y0) > precision:
        x0 = x1
        y0 = y1
        delx, dely = grad(x0, y0)
        a = alpha(x0, y0, delx, dely)
        print(f'Alpha: {a}')
        x1 = x0 + a*delx
        y1 = y0 + a*dely
        iteracoes += 1
    
    return x0, y0, f(x0, y0), iteracoes



def gradient(x, y):
    return [6*
    x + 3*y + 1, 3*x + 4*y + 1]

def function(x, y):
    return (3 * (x**2)) + (3 * x * y) + (2 * (y**2)) + x + y


x,y,z,interacoes = minimize_2var_func_V2(0, 0, function, gradient, 10**(-5))
print(f'Valor do Mínimo de f(x): ({x}; {y}; {z}) - Em {interacoes} Interações')
```

    Alpha 1: 0.5
    Alpha 2: 0.125
    Alpha 3: 0.5
    Alpha 4: 0.125
    Alpha 5: 0.5
    Alpha 6: 0.125
    Alpha 7: 0.5
    Alpha 8: 0.125
    Valor do Mínimo de f(x): (-0.0666656494140625; -0.1999969482421875; -0.13333333330228925) - Em 9 Interações

