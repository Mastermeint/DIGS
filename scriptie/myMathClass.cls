\LoadClass{article}

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{myUvaClass}[2016/10/16 custom class intended for UvA mathworks]

\RequirePackage{amsmath, mathtools, amsfonts, amsthm, wrapfig, graphicx}

%for appendix
\RequirePackage{thmtools, nameref, cleveref}
\RequirePackage[toc,page]{appendix}

% Define important math sets
\newcommand{\C}{\mathbb{C}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\F}{\mathbb{F}}
\newcommand{\bigO}[1]{\mathcal{O}(#1)}

% Mathematical environments
\theoremstyle{plain}
\newtheorem{thm}{Stelling}[section]
\newtheorem{lem}[thm]{Lemma}
\newtheorem{prop}[thm]{Propositie}
\newtheorem{cor}{Gevolg}[thm]

\theoremstyle{definition}
\newtheorem{defn}{Definitie}[section]
\newtheorem{conj}{Conjecture}[section]
\newtheorem{exmp}{Example}[section]

\theoremstyle{remark}
\newtheorem*{rem}{Remark}
\newtheorem*{note}{Note}


%http://tex.stackexchange.com/questions/53698/how-do-i-put-a-circle-around-an-operator
\usepackage{wasysym}
\usepackage{tikz}
%The given symbol or text (\text{mytext}) in a circle
%To be used always in math mode
\newcommand{\circlesign}[1]{ 
    \mathbin{
        \mathchoice
        {\buildcirclesign{\displaystyle}{#1}}
        {\buildcirclesign{\textstyle}{#1}}
        {\buildcirclesign{\scriptstyle}{#1}}
        {\buildcirclesign{\scriptscriptstyle}{#1}}
    }
}
\newcommand\buildcirclesign[2]{%
    \begin{tikzpicture}[baseline=(X.base), inner sep=0, outer sep=0]
    \node[draw,circle] (X)  {\ensuremath{#1 #2}};
    \end{tikzpicture}%
}


