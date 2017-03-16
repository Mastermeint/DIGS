\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Backend/uvamath}[2014/01/20 0.2 Bachelor wiskunde LaTeX document class]
\RequirePackage{graphicx}

\DeclareOption{english}{
% English values for translatable constant text
\def\supervisorstext{Supervisors}
\def\titletext{Title}
\def\authortextsingular{Authors}
\def\authortextplural{Authors}
\def\datetext{Date}
\def\secondgradertext{Second graders}
}

% Default values for translatable constant text
\def\supervisorstext{Begeleiding}
\def\titletext{Titel}
\def\authortextsingular{Auteur}
\def\authortextplural{Auteurs}
\def\datetext{Einddatum}
\def\secondgradertext{Tweede beoordelaar}

\ProcessOptions

\LoadClassWithOptions{scrreprt}
\setkomafont{disposition}{\bfseries}
%\RequirePackage[a4paper]{geometry}


% Variable text commands
\def\institute#1{\gdef\@institute{#1}}
\def\instituteaddress#1{\gdef\@instituteaddress{#1}}
\def\coverimage#1{\gdef\@coverimage{#1}}
\def\what#1{\gdef\@what{#1}}
\def\supervisors#1{\gdef\@supervisors{#1}}
\def\secondgrader#1{\gdef\@secondgrader{#1}}

% Default values for variable text
\institute{
Korteweg-de Vries Instituut voor Wiskunde\\\smallskip
Faculteit der Natuurwetenschappen, Wiskunde en Informatica\\\smallskip
Universiteit van Amsterdam\\\bigskip
\includegraphics[width=0.070\hsize]{img/uvalogo}
}
\instituteaddress{Korteweg-de Vries Instituut voor Wiskunde\\
Universiteit van Amsterdam\\
Science Park 904, 1098 XH Amsterdam\\
\url{http://www.science.uva.nl/math}{}
}
\coverimage{}
\what{}
\supervisors{\relax}
\secondgrader{\relax}

% Author command
\newcounter{authors}
\def\@authordetails{}
\def\@authors{}
\renewcommand\author[2][\relax]{
    \ifnum\value{authors}>0\relax
        \addtocounter{authors}{1}
        \xdef\@authors{\@authors, #2}
        \protected@xdef\@authordetails{\@authordetails\protect{\\}\indent #2\if#1\relax\else{, #1}\fi}
    \else
        \addtocounter{authors}{1}
        \gdef\@authors{#2}
        \gdef\@authordetails{#2, #1}
    \fi
}



% Maketitle command
\renewcommand{\maketitle}{
\pagestyle{empty}
\vspace*{30pt}
{\centering
{\LARGE \textbf{\bf \@title}\par}\bigskip
{\Large \@authors
\par}\bigskip
{\Large \@date\par}\bigskip\bigskip
{\large \@what\par}\medskip
\if\@supervisors\relax\else {\large\supervisorstext: \@supervisors\par}\bigskip\fi
\vspace{\stretch{1}}
\@coverimage
\vspace{\stretch{1}}

\@institute

\vspace{-70pt}
}
\newpage
\pagestyle{plain}
}

% Abstract and info page
\renewenvironment{abstract}{\section*{\abstractname}}{
\vfill
%\section*{Gegevens}
\noindent
\titletext: \@title\\
\ifnum\value{authors} < 2%
\authortextsingular: \@authordetails\\
\else
\authortextplural:\\ \indent\@authordetails\\
\fi
\if\@supervisors
    \relax
\else 
    \supervisorstext: \@supervisors\\ 
\fi
\if\@secondgrader
    \relax
\else 
    \secondgradertext: \@secondgrader\\
\fi
\datetext: \@date \\
\bigskip\noindent
\@instituteaddress
\newpage
}

\endinput

